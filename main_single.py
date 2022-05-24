import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

from Dataset import SingleSequenceLoader
from Model import Single_CLassifier
from Utils import plotting, visualization

import numpy as np

path = "/home/abulubad/Graduation-Project/Data"
model = Single_CLassifier()
batch_size = 1
model.train_Classifier_only(trainGen=SingleSequenceLoader(
    path, batch_size, 0.1),
                            valGen=SingleSequenceLoader(path,
                                                        batch_size,
                                                        0.1,
                                                        is_val=True),
                            batch_size=batch_size,
                            Nepoch=2)
