#%%
import os
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps
#%%

#Figure out a way how to segment foreground from background
#Create Object Detection Dataloader 
#Implement anomaly detection for bounding boxes on MLO and CC view
#Implement CC and MLO models
#Implement Image Classifications and Object detection (anomaly on Object detection)


class Dataset(Sequence):
    def __init__(self, path, view="CC", is_val=False, batch_size=5, classification=True, input_shape=255):
        #Largest dimensions 3481, 2746
        #Average dimensions 2533, 1588
        
        self.path = path 
        self.is_val = is_val
        self.batch_size = batch_size
        self.classification = classification
        self.input_shape = input_shape
        self.view = view
        if self.classification:
            if self.view != "CC" and self.view != "MLO":
                raise AttributeError(f"There is no such view as {self.view}, please either use \"MLO\" or \"CC\"")
            else:
                self.img, self.label = self.ClassificationImageLoad()
        self.idx = []
        self.numimg = len(self.img)
        if is_val:
            self.idx = [i for i in range(0,self.num_img)]
            
        
    def __getitem__(self, index):
        start = index*self.batch_size
        end = index*self.batch_size + self.batch_size
        tempList = self.idx[start:end]
        img = [self.img[i] for i in tempList]
        label = [self.label[i] for i in tempList]
        
    def __len__(self):
        return int(np.floor(len(self.idx)/self.batch_size))




    def ClassificationImageLoad(self):
        Images_Array = []
        status = []
        desired_size = (255,400)
        xlsx = pd.read_excel(self.path + "\DataWMask.xlsx")
        if self.view == "CC":
            for i in range(0,len(xlsx),2):
                img_path = xlsx['fullPath'][i]
                image = load_img(r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\Dataset\MINI-DDSM-Complete-JPEG-8\\" + img_path, grayscale=True)
                image = self.resize_with_padding(image,desired_size)
                Images_Array.append(image)
                if xlsx['Tumour_Contour'][i] == "Benign" or "Cancer":
                    if xlsx['Tumour_Contour'][i] == '-':
                        status.append('Normal')
                    else:
                        status.append(xlsx['Tumour_Contour'][i])
                else:
                    status.append('Normal')
        if self.view == "MLO":
            for i in range(1,len(xlsx),2):

                img_path = xlsx['fullPath'][i]
                image = load_img(r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\Dataset\MINI-DDSM-Complete-JPEG-8\\" + img_path, grayscale=True)
                image = self.resize_with_padding(image,desired_size)
                Images_Array.append(image)
                if xlsx['Tumour_Contour'][i] == "Benign" or "Cancer":
                    if xlsx['Tumour_Contour'][i] == '-':
                        status.append('Normal')
                    else:
                        status.append(xlsx['Tumour_Contour'][i])
                else:
                    status.append('Normal')

        return Images_Array, status

    def resize_with_padding(self, image, desired_size=(255, 400)):
        image.thumbnail(desired_size)
        d_width = desired_size[0] - image.size[0]
        d_height = desired_size[1] - image.size[1]
        pad_width = d_width // 2 
        pad_height = d_height //2
        padding = (pad_width, pad_height, d_width-pad_width, d_height-pad_height)
        return ImageOps.expand(image,padding)

#%%
folder_path = r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\Code\Dataset\MINI-DDSM-Complete-JPEG-8"
#cc, mlo, label = Dataset(folder_path)
# %%
image_1 = load_img(r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\Code\Dataset\MINI-DDSM-Complete-JPEG-8\Benign\0033\C_0033_1.LEFT_CC.jpg", grayscale=True)
#%%
image_np = np.array(image_1)
# %%
xlsx = pd.read_excel(folder_path + "\DataWMask.xlsx")
temp = np.array(load_img(os.path.join(r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\code\Dataset\MINI-DDSM-Complete-JPEG-8", xlsx['fullPath'][0]),grayscale=True)).shape
temp_lst = [temp[0],temp[1]]
for i in range(1,len(xlsx)-7000):
    img_path = os.path.join(r"C:\Users\yacou\Desktop\Studies\1. Deep Learning\GP\code\Dataset\MINI-DDSM-Complete-JPEG-8", xlsx['fullPath'][i])    
    image_x = load_img(img_path, grayscale=True)
    x = np.array(image_x).shape
    temp_lst.append(x)
    
print(temp_lst)
# %%
temp_lst.pop(0)
# %%
height = 0
width = 0
# %%
for i in temp_lst:
    height = height + i[0]
    width = width + i[1]

h_avg = height/(len(temp_lst)-1)
w_avg = width/(len(temp_lst)-1)

print(f"height = {h_avg}\nwidth = {w_avg}")

# %%
image_x.thumbnail()