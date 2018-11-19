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
from keras.layers import Activation, LeakyReLU, Lambda
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.models import Sequential

class AlexNet():
    
    def __init__(self, input_size):

        input_shape = (input_size[0],input_size[0],1)
        model = Sequential()
        model.add(Conv2D(64,input_shape=input_shape, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.35))

        model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=1./20))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.35))

        model.add(Flatten())

        model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

class Atten_custom_3_AlexNet(BaseFeatureExtractor):
    
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
        

        p = Lambda(self.permute_dim)(x)
        
        s = K.int_shape(p) # 512, 6, 6
        
        #A = Reshape((s[1]*s[2], s[3]))(x)
        A = Reshape((s[1], s[2]*s[3]))(p)
        A = Permute((2, 1))(A)
        
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
    def permute_dim(self, x):
        return K.permute_dimensions(x, (0,3,1,2))

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



def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
  
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape = (48,48,1), classes = 7):

    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((1, 1))(X_input)
    
    # Stage 1
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    """
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    """
    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2))(X)
    print(K.int_shape(X))
    # output layer
    X = Flatten()(X)
    x = Dense(128, activation='relu')(X)
    out = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = out, name='ResNet')

    return model