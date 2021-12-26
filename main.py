from Dataset import DataSequenceLoader
from Model import EFFNET
path = r"C:\Users\yacou\Desktop\GP\Code\Dataset\JPEG-Trial"
model = EFFNET()

#model.train_Classifier_only(trainGen=DataSequenceLoader(path,1), valGen=DataSequenceLoader(path,1,is_val=True))

trainGen=DataSequenceLoader(path,1)
print(trainGen[:])


