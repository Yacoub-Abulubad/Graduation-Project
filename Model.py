import os
from tensorflow.keras import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import EfficientNetB0 as Effnet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization

class EFFNET:
    def __init__(self):

        input_img = Input(shape=(252,252))
        Backbone = Effnet(include_top=False, weights= 'imagenet', input_tensor=input_img)
        Backbone.trainable = False

        x = MaxPooling2D((2, 2), padding='valid')(Backbone.output)
        x = Flatten(name="avg_pool")(x)
        x = BatchNormalization(name='BN_CL')(x)
        #x = Dense(640, activation = 'relu')(x) #Added Dense Layer
        #x = Dense(300, activation = 'relu')(x) #Added Dense Layer
        x = Dense(10, activation = 'relu')(x) #Added Dense Layer
        x = Dropout(0.2)(x)
        xFC = Dense(3, activation='softmax',name="FC")(x)
        self.model= Model(inputs= input_img, outputs=[xFC])
    
    def fix_Backbone_weights(self):

      for i,layer in enumerate(self.model.layer):
        if layer.name == 'BN_CL' or layer.name == "FC" or layer.name == "avg_pool":
          self.model.layers[i].trainable= True
        else:
          self.model.layers[i].trainable= False

    def unfix_layers_Backbone_weights(self, Nlayer=3):

        for i,_ in enumerate(self.model.layers[-5-Nlayer:]):
            self.model.layers[i].trainable= True

    def train_Classifier_only(self,trainGen,valGen,Nepoch=1):
        callBack= EarlyStopping(
                                  monitor="val_recall",
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                              )
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
        #self.model.summary()
        self.model.compile(optimizer='adam', loss={"FC" :'categorical_crossentropy'}, metrics={"FC" :[Precision(), Recall()]})
        self.FC_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch,callbacks=[callBack,rlrop])

    def train_Classifier_withBackbone(self,trainGen,valGen,Nepoch=30,Nlayers=3):
        self.unfix_layers_Backbone_weights(Nlayer= Nlayers)

        callBack= EarlyStopping(
                                  monitor="val_recall_1",
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode="max",
                                  baseline=None,
                                  restore_best_weights=True,
                              )
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1,min_delta=0.05)
        self.model.summary()
        self.model.compile(optimizer='adam', loss={"FC" :'categorical_crossentropy'}, metrics={"FC" :[Precision(), Recall()]})
        self.FC_B_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch,callbacks=[callBack,rlrop])
      
    def save_model(self,path,modelname="Effnet_Model"):
        self.model.save(os.path.join(path,modelname)+".h5")

    def save_weights(self,path,modelname="Effnet_Model_weights"):
        self.model.save_weights(os.path.join(path,modelname)+".h5")

    def load_weights(self,path,checkpoint="Effnet_Model_weights"):
        self.model.load_weights(os.path.join(path,checkpoint)+".h5")
