#%%
from Dataset import DataSequenceLoader
from Model import EFFNET
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

path = r"C:\Users\yacou\Desktop\GP\Code\Graduation-Project\Data"
model = EFFNET()

model.train_Classifier_only(trainGen=DataSequenceLoader(path,2), valGen=DataSequenceLoader(path,2,is_val=True))

# %%
