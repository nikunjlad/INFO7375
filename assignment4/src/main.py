import torch, warnings, torchvision, os, h5py, time, yaml, datetime, logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, Dataset, sampler, SubsetRandomSampler, TensorDataset
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')



class Main:

    def __init__(self):

        os.chdir("..")   # change to base path
        # loading the YAML configuration file
        with open("config.yaml", 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d  %H.%M")
        os.chdir("logs")   # change to logs directory
        # getting the custom logger
        self.logger_name = "resnets_" + self.current_time + "_.log"
        self.logger = self.get_loggers(self.logger_name)
        self.logger.info("Classification using ResNets!")
        self.logger.info("Current time: " + str(self.current_time))
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
    @staticmethod
    def configure_cuda(device_id):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('CUDA is not available. Training on CPU ...')
        else:
            torch.cuda.set_device(device_id)
            print('CUDA is available! Training on Tesla T4 Device {}'.format(str(torch.cuda.current_device())))

    def load_data(self, path):
        # loading data
        hf = h5py.File(path, 'r')
        # train, test data with labels being converted to numpy array from HDF5 format
        x_train = np.array(hf.get("X_train"), dtype=np.float32)
        y_train = np.array(hf.get("y_train"), dtype=np.int64)
        x_test = np.array(hf.get("X_test"), dtype=np.float32)
        y_test = np.array(hf.get("y_test"), dtype=np.int64)
        print("Training data: ", x_train.shape)
        print("Training labels: ", y_train.shape)
        print("Testing data: ", x_test.shape)
        print("Testing labels: ", y_test.shape)
        
        return x_train, y_train, x_test, y_test

    def main(self):

        # configure GPU if available
        if self.config["HYPERPARAMETERS"]["GPU"]:
            if self.config["HYPERPARAMETERS"]["DEVICE"] is not None:
                self.configure_cuda(self.config["HYPERPARAMETERS"]["DEVICE"])

        # loading data
        data_path = os.path.join(self.config["DATALOADER"]["DATA_DIR"], "data/cifar-10/cifar10.h5")
        x_train, y_train, x_test, y_test = self.load_data(data_path)

if __name__ == "__main__":
    m = Main()
    m.main()

