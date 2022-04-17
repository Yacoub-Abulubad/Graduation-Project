#%%
from Dataset import DataSequenceLoader
from Model import EFFNET
from Utils.visualization import visualize_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")

path = "/home/abulubad/Graduation-Project/Data"
model = EFFNET()
#%%
visualize_model(model)

#%%
model.train_Classifier_only(trainGen=DataSequenceLoader(path, 3),
                            valGen=DataSequenceLoader(path, 3, is_val=True))
