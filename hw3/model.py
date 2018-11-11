import numpy as np
import pandas as pd
import sys
#import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, LeakyReLU
from keras.initializers import glorot_uniform
import keras.backend as K

class AlexNet():
    
    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        # layer 1
        x = Conv2D(64,kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal')(input_image)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)

        # layer 2
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)

        # layer 3
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)

        x = Flatten()(x)

        x = Dense(512, kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512,  kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(7, activation='softmax', kernel_initializer='glorot_normal')(x)
        
        model = Model(input_image, x)

        self.model = model

class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

    def softmax(x, axis=1):
        print('x', K.int_shape(x))
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

class Atten_AlexNet(BaseFeatureExtractor):
    
    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        # layer 1
        x = Conv2D(64,kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal')(input_image)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)
        print(K.int_shape(x))
        
        # layer 2
        x = Conv2D(128,kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)
        print(K.int_shape(x))
        
        # layer 3
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)
        print(K.int_shape(x))
        # layer 4
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)
        print(K.int_shape(x))
        print(type(x))
        s = K.int_shape(x)
        A = Reshape((s[1]*s[2], s[3]))(x)
        #A = Permute((2, 1))(A)
        print('A', K.int_shape(A))
        e = Dense(64, activation = "tanh")(A)
        energies = Dense(1, activation = "relu")(e)
        print('energies', K.int_shape(energies))
        alphas = Activation(K.softmax, name='attention_weights')(energies)


        attention = Dot(axes = 1)([alphas, A])
        #attention = Multiply()([alphas, A])
        print('attention', K.int_shape(attention))

        attention_flat = Flatten()(attention)

        print('attention_flat', K.int_shape(attention_flat))

        CNN_flat = Flatten()(x)
        print('CNN_flat', K.int_shape(CNN_flat))
        x = Concatenate()([attention_flat, CNN_flat])  
        print('Concatenate', K.int_shape(x))

        x = Dense(512, kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512,  kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(7, activation='softmax', kernel_initializer='glorot_normal')(x)
        
        model = Model(input_image, x)

        self.model = model

class Atten_3_AlexNet(BaseFeatureExtractor):
    
    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        # layer 1
        x = Conv2D(64,kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal')(input_image)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)
        print(K.int_shape(x))
        
        # layer 2
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)
        print(K.int_shape(x))
        # layer 3
        x = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=1./20)(x)
        
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.35)(x)
        print(K.int_shape(x))
        print(type(x))
        s = K.int_shape(x)
        A = Reshape((s[1]*s[2], s[3]))(x)
        #A = Permute((2, 1))(A)
        print('A', K.int_shape(A))
        e = Dense(64, activation = "tanh")(A)
        energies = Dense(1, activation = "relu")(e)
        print('energies', K.int_shape(energies))
        alphas = Activation(K.softmax, name='attention_weights')(energies)


        attention = Dot(axes = 1)([alphas, A])
        #attention = Multiply()([alphas, A])
        print('attention', K.int_shape(attention))

        attention_flat = Flatten()(attention)

        print('attention_flat', K.int_shape(attention_flat))

        CNN_flat = Flatten()(x)
        print('CNN_flat', K.int_shape(CNN_flat))
        x = Concatenate()([attention_flat, CNN_flat])  
        print('Concatenate', K.int_shape(x))

        x = Dense(512, kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512,  kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(7, activation='softmax', kernel_initializer='glorot_normal')(x)
        
        model = Model(input_image, x)

        self.model = model

        
