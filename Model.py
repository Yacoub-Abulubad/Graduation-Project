import os
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import EfficientNetB0 as Effnet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization


def hierarchy_pred(model_1, model_2, dataGen):
    """For performing predictions when using `class Gierarchy` model

    Args:
        model_1 (object): Object of the first model. This model will be responsible to classify between `Normal` and `Tumor`
        model_2 (object): Object of the second model. This model will be responsible to classify between `Benign` and `Cancer`
        dataGen (object): Object of validation data generator/loader

    Returns:
        List, List: `YY_preds` list of predicted classes. `YY_true` list of true classes
    """
    YY_preds = []
    YY_true = []
    for XX, YY in dataGen:
        YY_true.append(YY[0])
        YY_preds_first = model_1.model.predict(XX)
        if YY_preds_first[0][1] > YY_preds_first[0][0]:
            YY_preds_second = model_2.model.predict(XX)
            if YY_preds_second[0][1] > YY_preds_second[0][0]:
                YY_preds.append([0, 0, 1])
            else:
                YY_preds.append([0, 1, 0])
        else:
            YY_preds.append([1, 0, 0])
    YY_true = np.array(YY_true)
    YY_true.reshape((-1, 3))
    return YY_preds, YY_true


class Hierarchy:
    """Hierarchy classifier
    """

    def __init__(self):
        """Effnet backbone
        """
        input_img = Input(shape=(252, 252, 3))
        Backbone = Effnet(include_top=False,
                          weights='imagenet',
                          input_tensor=input_img)
        Backbone.trainable = False

        x = MaxPooling2D((2, 2), padding='valid')(Backbone.output)
        x = Flatten(name="avg_pool")(x)
        x = BatchNormalization(name='BN_CL')(x)
        x = Dense(10, activation='relu')(x)
        x = Dropout(0.2)(x)
        xFC = Dense(2, activation='softmax', name="FC")(x)
        self.model = Model(inputs=input_img, outputs=[xFC])

    def fix_Backbone_weights(self):
        """To fix the backbone weights
        """
        for i, layer in enumerate(self.model.layer):
            if layer.name == 'BN_CL' or layer.name == "FC" or layer.name == "avg_pool":
                self.model.layers[i].trainable = True
            else:
                self.model.layers[i].trainable = False

    def unfix_layers_Backbone_weights(self, Nlayer=3):
        """To unfix the backbone weights

        Args:
            Nlayer (int, optional): the amount of layers to unfix from the end. Defaults to 3.
        """
        for i, _ in enumerate(self.model.layers[-5 - Nlayer:]):
            self.model.layers[i].trainable = True

    def train_Classifier_only(self, trainGen, valGen, batch_size, Nepoch=1):
        """To only train the FC layer of the network

        Args:
            trainGen (class): class of training data
            valGen (class): class of validation data
            Nepoch (int, optional): for how many epochs to iterrate. Defaults to 1.
        """
        callBack = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                                  mode='min',
                                  patience=3,
                                  factor=0.5,
                                  min_lr=1e-6,
                                  verbose=1,
                                  min_delta=0.05)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss={"FC": 'categorical_crossentropy'},
                           metrics={"FC": [Precision(), Recall()]})
        self.FC_history = self.model.fit(trainGen,
                                         batch_size=batch_size,
                                         validation_data=valGen,
                                         epochs=Nepoch,
                                         callbacks=[callBack, rlrop])

    def train_Classifier_withBackbone(self,
                                      trainGen,
                                      valGen,
                                      Nepoch=30,
                                      Nlayers=3,
                                      lr=0.0005):
        """To train FC with the backbone

        Args:
            trainGen (class): class of training data
            valGen (class): class of validation data
            Nepoch (int, optional): for how many epochs to iterrate. Defaults to 1.
            Nlayer (int, optional): the amount of layers to unfix from the end. Defaults to 3.
        """
        self.unfix_layers_Backbone_weights(Nlayer=Nlayers)

        callBack = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                                  mode='min',
                                  patience=3,
                                  factor=0.5,
                                  min_lr=1e-6,
                                  verbose=1,
                                  min_delta=0.05)
        self.model.summary()
        optimizer = Adam(lr=lr)
        self.model.compile(optimizer=optimizer,
                           loss={"FC": 'categorical_crossentropy'},
                           metrics={"FC": [Precision(), Recall()]})
        self.FC_B_history = self.model.fit(trainGen,
                                           validation_data=valGen,
                                           epochs=Nepoch,
                                           callbacks=[callBack, rlrop])

    def save_model(self, path, modelname="MobileNet_Model"):
        """To save the whole model

        Args:
            path (str): the path to where the model should be saved
            modelname (str, optional): The model name. Defaults to "MobileNet_Model".
        """
        self.model.save(os.path.join(path, modelname) + ".h5")

    def save_weights(self, path, checkpoint="MobileNetV2_Model_weights"):
        """To save the model weights

        Args:
            path (str): the path to where the weights should be saved
            checkpoint (str, optional): The checkpoint name. Defaults to "MobileNetV2_Model_weights".
        """
        self.model.save_weights(os.path.join(path, checkpoint) + ".h5")

    def load_weights(self, path, checkpoint="MobileNetV2_Model_weights"):
        """To load the saved weights

        Args:
            path (str): the path from where the weights should be loaded
            checkpoint (str, optional): The checkpoint name. Defaults to "MobileNetV2_Model_weights".
        """
        self.model.load_weights(os.path.join(path, checkpoint) + ".h5")


class Single_CLassifier:
    """A class to create a pretrained model
    """

    def __init__(self):
        """initialization to create a model class
        """
        input_img = Input(shape=(252, 252, 3))
        Backbone = Effnet(include_top=False,
                          weights='imagenet',
                          input_tensor=input_img)
        Backbone.trainable = False

        x = MaxPooling2D((2, 2), padding='valid')(Backbone.output)
        x = Flatten(name="avg_pool")(x)
        x = BatchNormalization(name='BN_CL')(x)
        x = Dense(10, activation='relu')(x)
        x = Dropout(0.2)(x)
        xFC = Dense(3, activation='softmax', name="FC")(x)
        self.model = Model(inputs=input_img, outputs=[xFC])

    def fix_Backbone_weights(self):
        """To fix the backbone weights
        """
        for i, layer in enumerate(self.model.layer):
            if layer.name == 'BN_CL' or layer.name == "FC" or layer.name == "avg_pool":
                self.model.layers[i].trainable = True
            else:
                self.model.layers[i].trainable = False

    def unfix_layers_Backbone_weights(self, Nlayer=3):
        """To unfix the backbone weights

        Args:
            Nlayer (int, optional): the amount of layers to unfix from the end. Defaults to 3.
        """
        for i, _ in enumerate(self.model.layers[-5 - Nlayer:]):
            self.model.layers[i].trainable = True

    def train_Classifier_only(self, trainGen, valGen, batch_size, Nepoch=1):
        """To only train the FC layer of the network

        Args:
            trainGen (class): class of training data
            valGen (class): class of validation data
            Nepoch (int, optional): for how many epochs to iterrate. Defaults to 1.
        """
        callBack = EarlyStopping(
            monitor="val_recall",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                                  mode='min',
                                  patience=3,
                                  factor=0.5,
                                  min_lr=1e-6,
                                  verbose=1,
                                  min_delta=0.05)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss={"FC": 'categorical_crossentropy'},
                           metrics={"FC": [Precision(), Recall()]})
        self.FC_history = self.model.fit(trainGen,
                                         batch_size=batch_size,
                                         validation_data=valGen,
                                         epochs=Nepoch,
                                         callbacks=[callBack, rlrop])

    def train_Classifier_withBackbone(self,
                                      trainGen,
                                      valGen,
                                      Nepoch=30,
                                      Nlayers=3,
                                      lr=0.0005):
        """To train FC with the backbone

        Args:
            trainGen (class): class of training data
            valGen (class): class of validation data
            Nepoch (int, optional): for how many epochs to iterrate. Defaults to 1.
            Nlayer (int, optional): the amount of layers to unfix from the end. Defaults to 3.
        """
        self.unfix_layers_Backbone_weights(Nlayer=Nlayers)

        callBack = EarlyStopping(
            monitor="val_recall_1",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )
        rlrop = ReduceLROnPlateau(monitor='val_loss',
                                  mode='min',
                                  patience=3,
                                  factor=0.5,
                                  min_lr=1e-6,
                                  verbose=1,
                                  min_delta=0.05)
        self.model.summary()
        optimizer = Adam(lr=lr)
        self.model.compile(optimizer=optimizer,
                           loss={"FC": 'categorical_crossentropy'},
                           metrics={"FC": [Precision(), Recall()]})
        self.FC_B_history = self.model.fit(trainGen,
                                           validation_data=valGen,
                                           epochs=Nepoch,
                                           callbacks=[callBack, rlrop])

    def save_model(self, path, modelname="Effnet_Model"):
        """To save the whole model

        Args:
            path (str): the path to where the model should be saved
            modelname (str, optional): The model name. Defaults to "Effnet_Model".
        """
        self.model.save(os.path.join(path, modelname) + ".h5")

    def save_weights(self, path, checkpoint="Effnet_Model_weights"):
        """To save the model weights

        Args:
            path (str): the path to where the weights should be saved
            checkpoint (str, optional): The checkpoint name. Defaults to "Effnet_Model_weights".
        """
        self.model.save_weights(os.path.join(path, checkpoint) + ".h5")

    def load_weights(self, path, checkpoint="Effnet_Model_weights"):
        """To load the saved weights

        Args:
            path (str): the path from where the weights should be loaded
            checkpoint (str, optional): The checkpoint name. Defaults to "Effnet_Model_weights".
        """
        self.model.load_weights(os.path.join(path, checkpoint) + ".h5")
