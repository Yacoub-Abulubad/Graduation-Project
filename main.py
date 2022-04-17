#%%
import os

from pydicom import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

from Dataset import DataSequenceLoader
from Model import EFFNET
#from Utils.visualization import visualize_model

path = "/home/abulubad/Graduation-Project/Data"
sheet = DataSequenceLoader(path, 3)
print(sheet.sheet)
model = EFFNET()
#%%
#visualize_model(model)

#%%
model.train_Classifier_only(trainGen=DataSequenceLoader(path, 3),
                            valGen=DataSequenceLoader(path, 3, is_val=True))

# %%
