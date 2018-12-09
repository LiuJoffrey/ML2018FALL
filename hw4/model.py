import numpy as np
import pandas as pd
import sys
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Embedding, BatchNormalization, Flatten, Conv2D, Reshape, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Concatenate, Permute, Dot, Input, LSTM, Multiply
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
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, LSTM
import keras.backend as K

class Bi_Lstm():
    def __init__(self, input_size, word_index, emb_size,embedding_matrix):
        input_s = Input(shape=(input_size,))
        x = Embedding(len(word_index)+1,output_dim= emb_size,
                            weights=[embedding_matrix],
                            input_length=input_size,
                            trainable=False)(input_s)
        x = Bidirectional(LSTM(units = 256, return_sequences = True))(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(0.5)(x)

        x = Bidirectional(LSTM(units = 512, return_sequences = True))(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Dropout(0.5)(x)

        e = Dense(64, activation='tanh')(x)
        energies = Dense(1, activation = "relu")(e)
        alphas = Activation(K.softmax, name='attention_weights')(energies)
        attention = Dot(axes = 1)([alphas, x])

        attention_flat = Flatten()(attention)

        x = Dense(256)(attention_flat)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        """
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        """
        out = Dense(1, activation='sigmoid')(x)

        model = Model(input_s, out)

        self.model = model


