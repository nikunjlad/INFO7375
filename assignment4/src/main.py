import torch, warnings, torchvision, os, h5py, time, yaml, datetime, logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, Dataset, sampler, SubsetRandomSampler, TensorDataset
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.DataGen import DataGen
from models.ResNet import ResNet

warnings.filterwarnings('ignore')


class Main(DataGen, ResNet):

    def __init__(self):
        # loading the YAML configuration file
        with open("config.yaml", 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d  %H.%M")
        os.chdir("logs")  # change to logs directory
        # getting the custom logger
        self.logger_name = "resnets_" + self.current_time + "_.log"
        self.logger = self.get_loggers(self.logger_name)
        self.logger.info("Classification using ResNets!")
        self.logger.info("Current time: " + str(self.current_time))
        self.train_on_gpu = False
        DataGen.__init__(self, self.config, self.logger)
        ResNet.__init__(self, self.config, self.logger)
        os.chdir("..")  # change directory to base path

    @staticmethod
    def get_loggers(name):
        logger = logging.getLogger("resnet")  # name the logger as squark
        logger.setLevel(logging.INFO)
        f_hand = logging.FileHandler(name)  # file where the custom logs needs to be handled
        f_hand.setLevel(logging.INFO)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        return logger

    # checking if cuda is available
    def configure_cuda(self, device_id):
        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu:
            print('CUDA is not available. Training on CPU ...')
        else:
            torch.cuda.set_device(device_id)
            print('CUDA is available! Training on Tesla T4 Device {}'.format(str(torch.cuda.current_device())))

    def main(self):

        # configure GPU if available
        if self.config["HYPERPARAMETERS"]["GPU"]:
            if self.config["HYPERPARAMETERS"]["DEVICES"] is not None:
                self.configure_cuda(self.config["HYPERPARAMETERS"]["DEVICES"][0])

        # loading data
        data_path = os.path.join(self.config["DATALOADER"]["DATA_DIR"], "data/cifar-10/cifar10.h5")
        self.load_data(data_path)
        self.split_data()
        self.configure_dataloaders()

        # define targets
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # get training, validation and testing dataset sizes and number of batches in each
        train_data_size = len(self.data["train_dataset"])
        valid_data_size = len(self.data["valid_dataset"])
        test_data_size = len(self.data["test_dataset"])
        num_train_data_batches = len(self.data["train_dataloader"])
        num_valid_data_batches = len(self.data["valid_dataloader"])
        num_test_data_batches = len(self.data["test_dataloader"])

        # display batch information
        print("Number of training samples: ", train_data_size)
        print("{} batches each having 64 samples".format(str(num_train_data_batches)))
        print("Number of validation samples: ", valid_data_size)
        print("{} batches each having 64 samples".format(str(num_valid_data_batches)))
        print("Number of testing samples: ", test_data_size)
        print("{} batches each having 64 samples".format(str(num_test_data_batches)))

        # export a subset of images
        batch = next(iter(self.data["train_dataloader"]))
        images, labels = batch

        grid = torchvision.utils.make_grid(images[:64], nrow=8)
        print(type(grid))
        plt.figure(figsize=(10, 10))
        np.transpose(grid, (1, 2, 0))
        save_image(grid, 'grid.png')
        for data, target in self.data["train_dataloader"]:
            print("Batch image tensor dimensions: ", data.shape)
            print("Batch label tensor dimensions: ", target.shape)
            break

        # select ResNet Model to train on
        net = self.ResNet18()

        # if training on GPU enabled, put the network on GPU
        if self.train_on_gpu:
            net = net.cuda()

        # if parallel configuration selected, train on the list of GPU's provided in a distributed fashion
        if self.config["HYPERPARAMETERS"]["PARALLEL"]:
            net = torch.nn.DataParallel(net, device_ids=self.config["HYPERPARAMETERS"]["DEVICES"])
            cudnn.benchmark = True

        # define loss criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=self.config["HYPERPARAMETERS"]["LR"], momentum=0.9, weight_decay=5e-4)

        # training and validation loop
        epochs = self.config["HYPERPARAMETERS"]["EPOCHS"]
        history = list()
        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            # Set to training mode
            net.train()

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0

            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(self.data["train_dataloader"]):

                if self.train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Clean existing gradients
                optimizer.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(),
                                                                                              acc.item() * 100))

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                net.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(self.data["valid_dataloader"]):

                    if self.train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # Forward pass - compute outputs on input data using the model
                    outputs = net(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                               loss.item(),
                                                                                                               acc.item() * 100))

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / float(train_data_size)

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / float(valid_data_size)

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()

            print("Epoch : {:03d}, Training: Loss: {:.4f}, \
                    Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, \
                    Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss,
                                                             avg_train_acc * 100, avg_valid_loss,
                                                             avg_valid_acc * 100, epoch_end - epoch_start))
        print(net)

        # save the model after training
        torch.save(net.state_dict(), 'resnet50.pt')  # save the resnet model
        hist = np.array(history)  # convert history from list to numpy array

        # training and validation loss curves
        plt.figure(figsize=(12, 12))
        x = [i for i in range(0, epochs)]
        plt.plot(x, hist[:, 0])
        plt.plot(x, hist[:, 1])
        plt.legend(['train_loss', 'valid_loss'], loc='upper right')
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("CIFAR-10 Loss Curves")
        plt.xlim(0, 50)
        fig = plt.gcf()
        fig.savefig("train_valid_loss.png")

        # training and validation accuracy curves
        plt.figure(figsize=(12, 12))
        x = [i for i in range(0, epochs)]
        plt.plot(x, hist[:, 2])
        plt.plot(x, hist[:, 3])
        plt.legend(['train_acc', 'valid_acc'], loc='upper right')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 Accuracy Curves")
        plt.xlim(0, 50)
        fig = plt.gcf()
        fig.savefig("train_valid_accuracy.png")

        # load model after training for testing
        net.load_state_dict(torch.load('resnet.pt'))

        test_loss = 0
        test_acc = 0

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            net.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(self.data["test_dataloader"]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                print(predictions.cpu().numpy()[0])
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                test_acc += acc.item() * inputs.size(0)

                print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(),
                                                                                               acc.item() * 100))

            avg_test_loss = test_loss / test_data_size
            avg_test_acc = test_acc / float(test_data_size)

            print("Test: Loss : {:.4f}, Accuracy: {:.4f}%".format(avg_test_loss, avg_test_acc * 100))

if __name__ == "__main__":
    m = Main()
    m.main()
