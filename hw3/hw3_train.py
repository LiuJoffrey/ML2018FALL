import numpy as np
import pandas as pd
import sys
#import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
import keras.backend as K
from model import *
arg = sys.argv
print(arg)


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


#raw = pd.read_csv(arg[1])
#label = raw.iloc[:,0].values
#print(label.shape)
#label = to_categorical(label)
#print(label.shape)
#np.save("reshape_label", label)
#data = raw.iloc[:,1]
#data = data.str.split(expand = True).astype('float32').values
#data = np.reshape(data, (-1,48,48,1))
#print(data.shape)
#np.save("reshape_data", data)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))

data = np.load("reshape_data.npy")
label = np.load("reshape_label.npy")
print(data.shape)
print(label.shape)
data = data /255.
"""
randomize = np.arange(len(data))
np.random.shuffle(randomize)
data = data[randomize]
label = label[randomize]
"""
val_data = data[27000:]
val_label = label[27000:]
data = data[:27000]
label = label[:27000]


datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=[0.8, 1.2],
                                shear_range=0.2,
                                horizontal_flip=True)
#model = ResNet50(input_shape = data.shape[1:], classes = 7)
#alexnet = Atten_AlexNet(data.shape[1:])
#model = alexnet.model
alexnet = Atten_3_AlexNet(data.shape[1:])
model = alexnet.model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

batch_size = 128
epochs = 400

#lr_reducer = ReduceLROnPlateau(factor=0.9, cooldown=0, patience=3, min_lr=10**(-5))
checkpointer = ModelCheckpoint(filepath='model'+'Atten_3_Alexnet'+'_{epoch:05d}_{val_acc:.5f}.h5', save_best_only=True,period=1,monitor='val_acc')
history = model.fit_generator(datagen.flow(data, label, batch_size=batch_size), steps_per_epoch=5*data.shape[0]//batch_size, epochs=epochs, validation_data=(val_data, val_label), callbacks=[checkpointer])

out = model.evaluate(data, label)
print(out)

out = model.evaluate(val_data, val_label)
print(out)






"""

    l = np.argmax(label[i])
    if l == 0:
        print("anger")
    elif l == 1:
        print("disgust")
    elif l == 2:
        print("fear")
    elif l == 3:
        print("happy")
    elif l == 4:
        print("sad")
    elif l == 5:
        print("surprise")
    elif l == 6:
        print("neutral")
    

    cv2.imshow("cc", data[i])
    cv2.waitKey(0)
"""


