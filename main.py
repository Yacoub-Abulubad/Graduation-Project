#%%
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

from Dataset import DataSequenceLoader
from Model import EFFNET
#from Utils.visualization import visualize_model

path = "/home/abulubad/Graduation-Project/Data"
sheet = DataSequenceLoader(path, 3)
model = EFFNET()
#%%
#visualize_model(model)

#%%
model.train_Classifier_only(trainGen=DataSequenceLoader(path, 1),
                            valGen=DataSequenceLoader(path, 1, is_val=True),
                            batch_size=1)

# %%
