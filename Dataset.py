from tensorflow.keras.preprocessing.image import load_img
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import Sequence
from skimage.measure import regionprops
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from PIL import ImageOps
from math import ceil
import pandas as pd
import numpy as np
import cv2
import os


class MSODSequenceLoader(Sequence):

    def __init__(self):
        pass


class ODSequenceLoader(Sequence):
    """For object detection

    Args:
        Sequence (class): inherting class
    """

    def __init__(self,
                 path,
                 batch_size=10,
                 data_size=0.5,
                 train_size=0.8,
                 is_val=False,
                 verbose=False):
        """Initialization

        Args:
            path (str): a string containing the location of the whole dataset dir
            batch_size (int): an integer to determine the batch size of the Data Sequence that will be input to the model for training
            data_size (float): a float between 1-0 determining the size of the data to use for training/validating
            train_size (float): a float between 1-0 determining the size of the training data to be used
            is_val (bool): a boolean to decide whether the needed data is for training or for validation
        """
        self.path = path
        self.data_size = data_size
        self.train_size = train_size
        self.batch_size = batch_size
        self.is_val = is_val
        self.verbose = verbose
        self.pathtolist()
        random = np.random.random_integers(0, 5000)
        self.idxList = [
            i
            for i in range(random, random + int(len(self.x_path) * data_size))
        ]

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
        np.random.seed(20)
        np.random.shuffle(self.idxList)
        print("shuffle done!")

    def pathtolist(self):
        """Convert the path to a set of paths to each image"""
        if self.verbose:
            print("Converting path to list!")
        x_paths = []
        y_paths = []
        sheet = pd.read_csv(self.path + r"/Dataset_Full_Full_Shuffled.csv")
        try:
            sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
        except:
            pass
        self.sheet = sheet
        count = 0
        self.Cancer = pd.DataFrame()
        self.Benign = pd.DataFrame()
        self.Normal = pd.DataFrame()
        mask_path = []

        if not self.is_val:
            for i in range(0, int(len(sheet) * self.train_size)):
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    y_paths.append(sheet['Status'][i])
                    self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    y_paths.append(sheet['Status'][i])
                    self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    count = count + 1
                    if count == 4:
                        y_paths.append('Normal')
                        self.sheet['Status'][i] = 'Normal'
                        self.Normal = self.Normal.append(sheet.loc[i])
                        count = 0

            self.Cancer = self.Cancer.reset_index(drop=True)
            self.Benign = self.Benign.reset_index(drop=True)
            self.Normal = self.Normal.reset_index(drop=True)
            balance_size = min(len(self.Cancer), len(self.Benign),
                               len(self.Normal))
            for i in range(0, balance_size):
                x_paths.append(self.Cancer['fullPath'][i])
                mask_path.append(self.Cancer['Tumour_Contour'][i])
                x_paths.append(self.Benign['fullPath'][i])
                mask_path.append(self.Benign['Tumour_Contour'][i])
                x_paths.append(self.Normal['fullPath'][i])

            self.y_path = y_paths
            self.x_path = x_paths
            self.mask_path = mask_path

        else:
            val_start = int(len(sheet) * self.data_size)

            for i in range(
                    val_start,
                    int(len(sheet) * (self.data_size +
                                      (1 - self.train_size)))):

                x_paths.append(self.path + sheet['fullPath'][i])
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    y_paths.append(sheet['Status'][i])
                    self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    y_paths.append(sheet['Status'][i])
                    self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    y_paths.append('Normal')
                    self.sheet['Status'][i] = 'Normal'
                    self.Normal = self.Normal.append(sheet.loc[i])

            #self.Cancer = self.Cancer.reset_index(drop=True)
            #self.Benign = self.Benign.reset_index(drop=True)
            #self.Normal = self.Normal.reset_index(drop=True)

            #for i in range(
            #        val_start,
            #        int(
            #            len(sheet) / 3 * (self.data_size +
            #                              (1 - self.train_size)))):
            #    x_paths.append(self.Cancer['fullPath'][i])
            #    x_paths.append(self.Benign['fullPath'][i])
            #    x_paths.append(self.Normal['fullPath'][i])

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
        if not self.is_val:
            return self.dataaug(x_batch)

        if self.verbose:
            print('Preprocessing images!')

        return np.asarray(x_batch)

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

    def dataaug(
            self,
            images):  #input should be a list of numpy arrays (list of images)
        """Perform image augmentation on the data


        Args:
           images (list): list of numpy arrays (list of images)

      Returns:
         array: numpy array
    """

        Auge = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
        out = Auge(images=images)
        return np.array(out)

    def vectorize(self, Y):
        """To vectorize (one hot encode) the labels

        Args:
            Y (list): list of strings of the 3 different cases

        Returns:
            array: an array of vectors representing different classes
        """
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

    def show_img(img):
        imgplot = plt.imshow(img)
        plt.show()

    def bb_calc(self):
        """calculates the bounding box region of the tumor
        """
        mask_path = self.mask_path
        for status in ['Normal', 'Benign', 'Cancer']:
            for path_mask_image in mask_path:
                #         print("path_mask_image", path_mask_image)
                if not os.path.exists(path_mask_image):
                    continue
                mask = cv2.imread(path_mask_image)
                mask[(mask > 0) & (mask < 255)] = 255

                cancer_one_channel = mask[:, :, 0]
                bboxs = regionprops(cancer_one_channel)

                for prop in bboxs:
                    bb = list(prop.bbox)
                    bb[0], bb[2] = bb[0] * scale_y, bb[2] * scale_y
                    bb[1], bb[3] = bb[1] * scale_x, bb[3] * scale_x
                    w = mask.shape[1] * scale_x
                    h = mask.shape[0] * scale_y
                    if True:
                        mask_rs = cv2.resize(mask,
                                             dsize=(CFG.IMG_SIZE,
                                                    CFG.IMG_SIZE),
                                             interpolation=cv2.INTER_AREA)
                        complete_rec = cv2.rectangle(img_main_rs,
                                                     (int(bb[1]), int(bb[0])),
                                                     (int(bb[3]), int(bb[2])),
                                                     (255, 0, 0), 2)
                        mask_rec = cv2.rectangle(mask_rs,
                                                 (int(bb[1]), int(bb[0])),
                                                 (int(bb[3]), int(bb[2])),
                                                 (255, 0, 0), 2)

                        self.show_img(img=complete_rec)
                        self.show_img(img=mask_rec)

                    center_x = (bb[1] + bb[3]) / (2 * w)
                    center_y = (bb[0] + bb[2]) / (2 * h)
                    height_norm = (bb[2] - bb[0]) / h
                    width_norm = (bb[3] - bb[1]) / w
                    if center_x > 1 or center_y > 1 or width_norm > 1 or height_norm > 1:
                        continue
                    yolo_txt.append("{} {} {} {} {}".format(mapping[status],\
                                                            str(np.round(center_x,4)),\
                                                            str(np.round(center_y,4)),\
                                                            str(np.round(width_norm,4)),\
                                                            str(np.round(height_norm,4))))
        yolo_txt = "\n".join(yolo_txt)
        yolo_txt = yolo_txt[:-1]
        file_txt.write(yolo_txt)
        file_txt.close()


class HierarchySequenceLoader(Sequence):
    """A data loader responsible for loading, preprocessing and feeding a batch of the data into the hierarchy model

    Args:
        Sequence (class): inherting class
    """

    def __init__(self,
                 path,
                 first_classifier=True,
                 batch_size=10,
                 data_size=0.5,
                 train_size=0.8,
                 is_val=False,
                 verbose=False):
        """Initialization

        Args:
            path (str): a string containing the location of the whole dataset dir
            batch_size (int): an integer to determine the batch size of the Data Sequence that will be input to the model for training
            data_size (float): a float between 1-0 determining the size of the data to use for training/validating
            train_size (float): a float between 1-0 determining the size of the training data to be used
            is_val (bool): a boolean to decide whether the needed data is for training or for validation
        """
        self.path = path
        self.data_size = data_size
        self.train_size = train_size
        self.batch_size = batch_size
        self.first_classifier = first_classifier
        self.is_val = is_val
        self.verbose = verbose
        self.pathtolist()
        random = np.random.random_integers(0, 5000)
        self.idxList = [
            i
            for i in range(random, random + int(len(self.x_path) * data_size))
        ]

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
        np.random.seed(20)
        np.random.shuffle(self.idxList)
        print("shuffle done!")

    def pathtolist(self):
        """Convert the path to a set of paths to each image"""
        if self.verbose:
            print("Converting path to list!")
        first_x_paths = []
        first_y_paths = []
        second_x_paths = []
        second_y_paths = []
        sheet = pd.read_csv(self.path + r"/Dataset_Full_Shuffled.csv")
        try:
            sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
        except:
            pass
        self.sheet = sheet
        count = 0
        self.Cancer = pd.DataFrame()
        self.Benign = pd.DataFrame()
        self.Normal = pd.DataFrame()

        if not self.is_val:
            for i in range(0, int(len(sheet) * self.train_size)):
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    first_y_paths.append('Tumor')
                    second_y_paths.append('Cancer')
                    self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    first_y_paths.append('Tumor')
                    second_y_paths.append('Benign')
                    self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    count = count + 1
                    if count == 2:
                        first_y_paths.append('Normal')
                        self.sheet['Status'][i] = 'Normal'
                        self.Normal = self.Normal.append(sheet.loc[i])
                        count = 0

            self.Cancer = self.Cancer.reset_index(drop=True)
            self.Benign = self.Benign.reset_index(drop=True)
            self.Normal = self.Normal.reset_index(drop=True)
            balance_size = min(len(self.Cancer), len(self.Benign),
                               len(self.Normal))
            for i in range(0, balance_size):
                first_x_paths.append(self.path + self.Cancer['fullPath'][i])
                first_x_paths.append(self.path + self.Benign['fullPath'][i])
                first_x_paths.append(self.path + self.Normal['fullPath'][i])
                second_x_paths.append(self.path + self.Cancer['fullPath'][i])
                second_x_paths.append(self.path + self.Benign['fullPath'][i])
            if self.first_classifier:
                self.y_path = first_y_paths
                self.x_path = first_x_paths
            else:
                self.y_path = second_y_paths
                self.x_path = second_x_paths

        else:
            val_start = int(len(sheet) * self.data_size)

            for i in range(
                    val_start,
                    int(len(sheet) * (self.data_size +
                                      (1 - self.train_size)))):

                first_x_paths.append(self.path + sheet['fullPath'][i])
                second_x_paths.append(self.path + sheet['fullPath'][i]))
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    first_y_paths.append('Tumor')
                    second_y_paths.append('Cancer')
                    #self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    first_y_paths.append('Tumor')
                    second_y_paths.append('Benign')
                    #self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    first_y_paths.append('Normal')
                    self.sheet['Status'][i] = 'Normal'
                    #self.Normal = self.Normal.append(sheet.loc[i])

            #self.Cancer = self.Cancer.reset_index(drop=True)
            #self.Benign = self.Benign.reset_index(drop=True)
            #self.Normal = self.Normal.reset_index(drop=True)

            #for i in range(
            #        val_start,
            #        int(
            #            len(sheet) / 3 * (self.data_size +
            #                              (1 - self.train_size)))):
            #    x_paths.append(self.Cancer['fullPath'][i])
            #    x_paths.append(self.Benign['fullPath'][i])
            #    x_paths.append(self.Normal['fullPath'][i])

            if self.first_classifier:
                self.y_path = first_y_paths
                self.x_path = first_x_paths
                print(self.y_path)
                print(self.x_path)
            else:
                self.y_path = second_y_paths
                self.x_path = second_x_paths

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
        if not self.is_val:
            return self.dataaug(x_batch)

        if self.verbose:
            print('Preprocessing images!')

        return np.asarray(x_batch)

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

    def dataaug(
            self,
            images):  #input should be a list of numpy arrays (list of images)
        """Perform image augmentation on the data


        Args:
           images (list): list of numpy arrays (list of images)

      Returns:
         array: numpy array
    """

        Auge = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
        out = Auge(images=images)
        return np.array(out)

    def vectorize(self, Y):
        """To vectorize (one hot encode) the labels for both indavidual classifiers

        Args:
            Y (list): list of strings of the 3 different cases

        Returns:
            array: an array of vectors representing different classes
        """
        first_classes = []
        second_classes = []
        batch_size = self.ending - self.start
        for i in range(batch_size):
            if Y[i] == 'Normal':
                first_classes.append(np.asarray([1, 0]))
            elif Y[i] == 'Cancer':
                first_classes.append(np.asarray([0, 1]))
                second_classes.append(np.asarray([0, 1]))
            else:
                first_classes.append(np.asarray([0, 1]))
                second_classes.append(np.asarray([1, 0]))
        if self.first_classifier:
            return first_classes
        else:
            return second_classes


class SingleSequenceLoader(Sequence):
    """A data loader responsible for loading, preprocessing and feeding a batch of the data into the model

    Args:
        Sequence (class): inherting class
    """

    def __init__(self,
                 path,
                 batch_size=10,
                 data_size=0.5,
                 train_size=0.8,
                 is_val=False,
                 verbose=False):
        """Initialization

        Args:
            path (str): a string containing the location of the whole dataset dir
            batch_size (int): an integer to determine the batch size of the Data Sequence that will be input to the model for training
            data_size (float): a float between 1-0 determining the size of the data to use for training/validating
            train_size (float): a float between 1-0 determining the size of the training data to be used
            is_val (bool): a boolean to decide whether the needed data is for training or for validation
        """
        self.path = path
        self.data_size = data_size
        self.train_size = train_size
        self.batch_size = batch_size
        self.is_val = is_val
        self.verbose = verbose
        self.pathtolist()
        random = np.random.random_integers(0, 5000)
        self.idxList = [
            i
            for i in range(random, random + int(len(self.x_path) * data_size))
        ]

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
        np.random.seed(20)
        np.random.shuffle(self.idxList)
        print("shuffle done!")

    def pathtolist(self):
        """Convert the path to a set of paths to each image"""
        if self.verbose:
            print("Converting path to list!")
        x_paths = []
        y_paths = []
        sheet = pd.read_csv(self.path + r"/Dataset_Full_Full_Shuffled.csv")
        try:
            sheet = sheet.loc[:, ~sheet.columns.str.contains('^Unnamed')]
        except:
            pass
        self.sheet = sheet
        count = 0
        self.Cancer = pd.DataFrame()
        self.Benign = pd.DataFrame()
        self.Normal = pd.DataFrame()

        if not self.is_val:
            for i in range(0, int(len(sheet) * self.train_size)):
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    y_paths.append(sheet['Status'][i])
                    self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    y_paths.append(sheet['Status'][i])
                    self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    count = count + 1
                    if count == 4:
                        y_paths.append('Normal')
                        self.sheet['Status'][i] = 'Normal'
                        self.Normal = self.Normal.append(sheet.loc[i])
                        count = 0

            self.Cancer = self.Cancer.reset_index(drop=True)
            self.Benign = self.Benign.reset_index(drop=True)
            self.Normal = self.Normal.reset_index(drop=True)
            balance_size = min(len(self.Cancer), len(self.Benign),
                               len(self.Normal))
            for i in range(0, balance_size):
                x_paths.append(self.Cancer['fullPath'][i])
                x_paths.append(self.Benign['fullPath'][i])
                x_paths.append(self.Normal['fullPath'][i])

            self.y_path = y_paths
            self.x_path = x_paths

        else:
            val_start = int(len(sheet) * self.data_size)

            for i in range(
                    val_start,
                    int(len(sheet) * (self.data_size +
                                      (1 - self.train_size)))):

                x_paths.append(self.path + sheet['fullPath'][i])
                if sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Cancer':
                    y_paths.append(sheet['Status'][i])
                    self.Cancer = self.Cancer.append(sheet.loc[i])
                elif sheet['Tumour_Contour'][i] != '-' and sheet['Status'][
                        i] == 'Benign':
                    y_paths.append(sheet['Status'][i])
                    self.Benign = self.Benign.append(sheet.loc[i])
                else:
                    y_paths.append('Normal')
                    self.sheet['Status'][i] = 'Normal'
                    self.Normal = self.Normal.append(sheet.loc[i])

            #self.Cancer = self.Cancer.reset_index(drop=True)
            #self.Benign = self.Benign.reset_index(drop=True)
            #self.Normal = self.Normal.reset_index(drop=True)

            #for i in range(
            #        val_start,
            #        int(
            #            len(sheet) / 3 * (self.data_size +
            #                              (1 - self.train_size)))):
            #    x_paths.append(self.Cancer['fullPath'][i])
            #    x_paths.append(self.Benign['fullPath'][i])
            #    x_paths.append(self.Normal['fullPath'][i])

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
        if not self.is_val:
            return self.dataaug(x_batch)

        if self.verbose:
            print('Preprocessing images!')

        return np.asarray(x_batch)

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

    def dataaug(
            self,
            images):  #input should be a list of numpy arrays (list of images)
        """Perform image augmentation on the data


        Args:
           images (list): list of numpy arrays (list of images)

      Returns:
         array: numpy array
    """

        Auge = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
        out = Auge(images=images)
        return np.array(out)

    def vectorize(self, Y):
        """To vectorize (one hot encode) the labels

        Args:
            Y (list): list of strings of the 3 different cases

        Returns:
            array: an array of vectors representing different classes
        """
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
