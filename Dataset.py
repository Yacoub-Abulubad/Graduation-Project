from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
from PIL import ImageOps
import pandas as pd
import numpy as np


class DataSequenceLoader(Sequence):
    def __init__(self, path, batch_size=10,train_size=0.8, is_val=False, verbose=False):
        """Initialization

        Args:
            - path (str): a string containing the location of the whole dataset dir
            - batch_size (int): an integer to determine the batch size of the Data Sequence that will be input to the model for training
            - train_size (float): a float between 1-0 determining the size of the training data to be used
            - is_val (bool): a boolean to decide whether the needed data is for training or for validation
        """
        self.path = path
        self.train_size = train_size
        self.batch_size = batch_size
        self.is_val = is_val
        self.verbose = verbose
        self.pathtolist()


    def __getitem__(self, index):
        """Load and pass one batch of images at a time per epoch to the model"""
        X_batch = self.x_path[index*self.batch_size: (index+1)*self.batch_size]
        Y_batch = self.y_path[index*self.batch_size: (index+1)*self.batch_size]
        
        return self.generate(X_batch), np.array(list(map(self.vectorize,Y_batch)))

    def __len__(self):
        """Measure the length of the dataset in batches"""
        return int(np.floor(len(self.x_path)/self.batch_size))

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
        sheet = pd.read_csv(self.path + r"\Dataset.csv")
        if not self.is_val:
            for i in range(0,int(len(sheet)*self.train_size)):
                x_paths.append(self.path + sheet['fullPath'][i])
            self.x_path = x_paths
            
            for i in range(0,int(len(sheet)*self.train_size)):
                if sheet['Tumour_Contour'][i]  != '-':
                    y_paths.append(sheet['Status'][i])
                else:
                    y_paths.append('Normal')
            self.y_path = y_paths

        else:
            for i in range(0,int(len(sheet)*(1-self.train_size))):
                x_paths.append(self.path + sheet['fullPath'][i])
            self.x_path = x_paths
            
            for i in range(0,int(len(sheet)*(1-self.train_size))):
                if sheet['Tumour_Contour'][i]  != '-':
                    y_paths.append(sheet['Status'][i])
                else:
                    y_paths.append('Normal')
            self.y_path = y_paths
        if self.verbose:
            print("Path to list conversion complete!")

    def generate(self, x_batch):
        #Load images
        #rescale and pad
        #preprocess
        x_array = np.empty_like(x_batch)
        if self.verbose:
            print('Generating and resizing images!')
        for i in range(len(x_batch)):
            temp = self.resize_with_padding(load_img(x_batch[i]))
            x_batch[i] = np.asarray(temp)
        if not self.is_val:
            x_batch = self.DataAugemntation(x_batch)
        if self.verbose:
            print('Preprocessing images!')
        
        return x_batch


    def resize_with_padding(self, image, desired_size=(252,252)):
        image.thumbnail(desired_size)
        d_width = desired_size[0] - image.size[0]
        d_height = desired_size[1] - image.size[1]
        pad_width = d_width // 2 
        pad_height = d_height //2
        padding = (pad_width, pad_height, d_width-pad_width, d_height-pad_height)
        return ImageOps.expand(image,padding)

    def DataAugemntation(self, images): #input should be a list of numpy arrays (list of images)
        Auge= iaa.RandAugment(n=(1,5),m=(3,15))
        Auge= iaa.RandAugment(n=(1,5),m=(10))
        out=Auge(images=images)
        return np.array(out)


    def vectorize(self, Y):
        print(Y)
        if Y == 'Normal':
            return [1,0,0]
        elif Y == 'Benign':
            return [0,1,0]
        else:
            return [0,0,1]
