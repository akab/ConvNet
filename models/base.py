import os

from abc import ABC, abstractmethod
from keras import Model
from keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        print('...epoch', epoch, 'end')


def load_weights_from_dir(check_dir):
    if os.path.exists(check_dir):
        files = os.listdir(check_dir)
        paths = [os.path.join(check_dir, basename) for basename in files if 'weights' in basename]
        if paths.__len__() == 1:
            return paths[0]
        elif paths.__len__() > 1:
            return max(paths, key=os.path.getctime)
        else:
            return None


class BaseNetwork(ABC):
    def __init__(self, name='', input_size=None):
        self.name = name
        self.input_size = input_size
        self.architecture = Model()

    @abstractmethod
    def define(self):
        """
        Define NN architecture
        :return: the model
        """
        pass

    @abstractmethod
    def train(self, train):
        """
        Train network on train data
        :param train: train data
        :return:
        """
        pass

    @abstractmethod
    def validate(self, test):
        """
        Validate network on test data
        :param test: test data
        :return:
        """
        pass
