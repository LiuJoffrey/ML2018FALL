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
"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))
"""

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
val_data = data[27500:]
val_label = label[27500:]
#data = data[:27500]
#label = label[:27500]
data = data
label = label

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                height_shift_range=0.2,
                                zoom_range=[0.8, 1.2],
                                shear_range=0.2,
                                horizontal_flip=True)

alexnet = AlexNet(data.shape[1:])
model = alexnet.model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()

batch_size = 128
epochs = 400

#lr_reducer = ReduceLROnPlateau(factor=0.9, cooldown=0, patience=3, min_lr=10**(-5))
checkpointer = ModelCheckpoint(filepath='model'+'Alexnet'+'_{epoch:05d}_{val_acc:.5f}.h5', save_best_only=True,period=1,monitor='val_acc')
history = model.fit_generator(datagen.flow(data, label, batch_size=batch_size), steps_per_epoch=1*data.shape[0]//batch_size, epochs=epochs, validation_data=(val_data, val_label), callbacks=[checkpointer])

out = model.evaluate(data, label)
print(out)

out = model.evaluate(val_data, val_label)
print(out)
model.save("model_Alexnet_final.h5")

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


