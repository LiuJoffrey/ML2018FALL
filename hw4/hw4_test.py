from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import sys
import jieba
import csv
from model import *
import re
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
import pickle
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

arg = sys.argv

test_csv = arg[1]
dict_big = arg[2]
out_file = arg[3]

jieba.set_dictionary(dict_big)
jieba.add_word('姜太公')
stop_punctuation = ['，',':',';','（','）','\'','。','-','「','」'] 

def jieba_lines(lines):
    c = 0
    cut_lines = []
    for line in lines:
        print("count: ", c)
        c+=1
        cut_line = jieba.lcut(line)
        for w_i in range(len(cut_line)):
            if cut_line[w_i] in stop_punctuation:
                cut_line[w_i] = " "
        cut_lines.append(cut_line)

    return cut_lines

def pre_processinog(lines):
	pre_lines = []
	for line in lines:
		for s in re.findall(r'[>+!?<."。， ]+', line):
			line = line.replace(s,s[0])
			
		pre_lines.append(line)
	return pre_lines


all_test = []

with open(test_csv) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        all_test.append(row['comment'])





all_test = pre_processinog(all_test)		
cut_test = jieba_lines(all_test)
np.save("cut_test", cut_test)
cut_test = np.load("cut_test.npy")
cut_test = cut_test.tolist()
print("finish jieba")
#embedding_model = Word2Vec.load("word2vec.model")	

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("load tokenizer")
sequences = tokenizer.texts_to_sequences(cut_test)
print("finish to sequence")
max_length = 100
test = pad_sequences(sequences, maxlen=max_length)
print("finish pad ")

# modelW2V-00002-0.72970.h5 0.72532
# modelW2V-00004-0.73410.h5 0.73647
# modelW2V-00003-0.76110.h5 0.75845



model = load_model("modelW2V-00003-0.76110.h5")
predict = model.predict(test, verbose=1, batch_size=80)
model = None
model0 = load_model("0_modelW2V-00004-0.75780.h5")
predict0 = model0.predict(test, verbose=1, batch_size=80)
model0 = None
model1 = load_model("new_modelW2V-00003-0.76187.h5")
predict1 = model1.predict(test, verbose=1, batch_size=80)
model0 = None
#model1 = load_model("1_modelW2V-00002-0.75650.h5")
#model2 = load_model("2_modelW2V-00003-0.75340.h5")
#model3 = load_model("3_modelW2V-00002-0.75800.h5")

print("finish load model")



#predict1 = model1.predict(test, verbose=1)
#predict2 = model2.predict(test, verbose=1)
#predict3 = model3.predict(test, verbose=1)

#predict = (predict + predict0 + predict1 + predict2)/4
predict = (predict + predict0+predict1)/3



out = []
for i in range(len(predict)):
    if predict[i] >= 0.5:
        out.append([str(i),1])
    else:
        out.append([str(i),0])
submission = open(out_file, "w+")
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(out)):
    s.writerow(out[i])
submission.close()




