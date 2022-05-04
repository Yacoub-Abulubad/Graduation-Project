from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
from PIL import ImageOps
from math import ceil
import pandas as pd
import numpy as np


class DataSequenceLoader(Sequence):
    """A data loader responsible for loading, preprocessing and feeding a batch of the data into the model

    Args:
        Sequence (class): inherting class
    """

    def __init__(self,
                 path,
                 batch_size=10,
                 train_size=0.8,
                 is_val=False,
                 verbose=False):
        """Initialization

        Args:
            path (str): a string containing the location of the whole dataset dir
            batch_size (int): an integer to determine the batch size of the Data Sequence that will be input to the model for training
            train_size (float): a float between 1-0 determining the size of the training data to be used
            is_val (bool): a boolean to decide whether the needed data is for training or for validation
        """
        self.path = path
        self.train_size = train_size
        self.batch_size = batch_size
        self.is_val = is_val
        self.verbose = verbose
        self.pathtolist()
        self.idxList = [i for i in range(len(self.x_path))]

    def __getitem__(self, index):
        """Load and pass one batch of images at a time per epoch to the model

        Args:
            index (int): the index of the idxList

        Returns:
            array: an array of a batch of images
        """
        self.start = index * self.batch_size
        self.ending = index * self.batch_size + self.batch_size
        if self.ending >= len(self.idxList):
            self.ending = len(self.idxList)
        self.X_batch = self.x_path[self.start:self.ending]
        self.Y_batch = self.y_path[self.start:self.ending]

        return self.generate(self.X_batch), np.array(
            self.vectorize(self.Y_batch))

    def __len__(self):
        """Measure the length of the dataset in batches"""
        return int(ceil(len(self.idxList) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle the list of paths after every epoch"""
        temp_list = list(zip(self.x_path, self.y_path))
        np.random.shuffle(temp_list)
        self.x_path, self.y_path = zip(*temp_list)
        print("Shuffle done!")

    def pathtolist(self):
        """Convert the path to a set of paths to each image"""
        if self.verbose:
            print("Converting path to list!")
        x_paths = []
        y_paths = []
        sheet = pd.read_csv(self.path + r"/Dataset_Full.csv")
        try:
            sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
        except:
            pass
        self.sheet = sheet
        if not self.is_val:
            for i in range(0, int(len(sheet) * self.train_size)):
                x_paths.append(self.path + sheet['fullPath'][i])
                if sheet['Tumour_Contour'][i] != '-':
                    y_paths.append(sheet['Status'][i])
                else:
                    y_paths.append('Normal')
            self.y_path = y_paths
            self.x_path = x_paths

        else:
            val_start = int(len(sheet) * self.train_size)
            for i in range(0, int(len(sheet) * (1 - self.train_size))):
                x_paths.append(self.path + sheet['fullPath'][val_start + i])
                if sheet['Tumour_Contour'][val_start + i] != '-':
                    y_paths.append(sheet['Status'][val_start + i])
                else:
                    y_paths.append('Normal')
            self.y_path = y_paths
            self.x_path = x_paths

        if self.verbose:
            print("Path to list conversion complete!")

    def generate(self, x_batch):
        """Loads, rescales, pads and preprocess the image here

        Args:
            x_batch (array): numpy array of a batch of images

        Returns:
            array: numpy array of a set of preprocesed and ready to use images
        """
        #Load images
        #rescale and pad
        #preprocess
        x_array = np.empty_like(x_batch)
        if self.verbose:
            print('Generating and resizing images!')
        for i in range(len(x_batch)):
            temp = self.resize_with_padding(load_img(x_batch[i]))
            x_batch[i] = np.asarray(temp)
        #if not self.is_val:
        #    x_batch = self.DataAugmentation(x_batch)
        if self.verbose:
            print('Preprocessing images!')

        return x_batch

    def resize_with_padding(self, image, desired_size=(252, 252)):
        """to resize images to desired size and add padding

        Args:
            image (array): a numpy array of an image
            desired_size (tuple, optional): the desired size of the image. Defaults to (252, 252).

        Returns:
            array: returns numpy array of resized image
        """
        image.thumbnail(desired_size)
        d_width = desired_size[0] - image.size[0]
        d_height = desired_size[1] - image.size[1]
        pad_width = d_width // 2
        pad_height = d_height // 2
        padding = (pad_width, pad_height, d_width - pad_width,
                   d_height - pad_height)
        return ImageOps.expand(image, padding)

    def DataAugmentation(
            self,
            images):  #input should be a list of numpy arrays (list of images)
        """Perform image augmentation on the data


        Args:
           images (list): list of numpy arrays (list of images)

      Returns:
         array: numpy array
    """

        Auge = iaa.RandAugment(n=(1, 5), m=(3, 15))
        Auge = iaa.RandAugment(n=(1, 5), m=(10))
        out = Auge(images=images)
        return np.array(out)

    def vectorize(self, Y):
        """To vectorize (one hot encode) the labels

        Args:
            Y (list): list of strings of the 3 different cases

        Returns:
            array: an array of vectors representing different classes
        """
        ###############################
        #Problem here
        ###############################
        classes = []
        batch_size = self.ending - self.start
        for i in range(batch_size):
            if Y[i] == 'Normal':
                classes.append(np.asarray([1, 0, 0]))
            elif Y[i] == 'Benign':
                classes.append(np.asarray([0, 1, 0]))
            else:
                classes.append(np.asarray([0, 0, 1]))

        return classes
