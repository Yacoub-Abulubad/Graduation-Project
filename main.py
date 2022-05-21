import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

from Dataset import DataSequenceLoader
from Model import EFFNET
from Utils import plotting

import numpy as np

path = "/home/abulubad/Graduation-Project/Data"
model = EFFNET()
batch_size = 8
model.train_Classifier_only(trainGen=DataSequenceLoader(path, batch_size, 0.1),
                            valGen=DataSequenceLoader(path,
                                                      batch_size,
                                                      0.1,
                                                      is_val=True),
                            batch_size=batch_size,
                            Nepoch=2)
