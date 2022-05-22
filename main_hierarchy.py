import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

from Dataset import HierarchySequenceLoader
from Model import Hierarchy
from Utils import plotting

import numpy as np

path = "/home/abulubad/Graduation-Project/Data"
first_model = Hierarchy()
second_model = Hierarchy()
batch_size = 1
first_model.train_Classifier_only(
    trainGen=HierarchySequenceLoader(path,
                                     first_classifier=True,
                                     batch_size=batch_size,
                                     data_size=0.1),
    valGen=HierarchySequenceLoader(path,
                                   first_classifier=True,
                                   batch_size=batch_size,
                                   data_size=0.1,
                                   is_val=True),
    batch_size=batch_size,
    Nepoch=2)

second_model.train_Classifier_only(
    trainGen=HierarchySequenceLoader(path,
                                     first_classifier=False,
                                     batch_size=batch_size,
                                     data_size=0.1),
    valGen=HierarchySequenceLoader(path,
                                   first_classifier=False,
                                   batch_size=batch_size,
                                   data_size=0.1,
                                   is_val=True),
    batch_size=batch_size,
    Nepoch=2)
