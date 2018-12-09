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

arg = sys.argv
data_csv = arg[1]
label_csv = arg[2]
dict_big = arg[3]

jieba.set_dictionary(dict_big)
jieba.add_word('姜太公')
stop_punctuation = ['，',':',';','（','）','\'','。','-','「','」'] #, '>', '<'
stop_word = []
"""
with open('stopword.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stop_word.append(data)


"""
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


def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1

def pre_processinog(lines):
	pre_lines = []
	for line in lines:
		for s in re.findall(r'[>+!?<."。， ]+', line):
			line = line.replace(s,s[0])
			
		pre_lines.append(line)
	return pre_lines
		
example = ["台大100%>>>>>>>>>>>>>科大  100%>>>>>>>>>>科大1%>>>>>>>>>你"]
example = pre_processinog(example)
print(example)

all_comment = []
all_label = []
all_test = []

with open(data_csv) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        all_comment.append(row['comment'])

with open(label_csv) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        all_label.append(float(row['label']))


all_label = np.array(all_label)

all_comment = pre_processinog(all_comment)


full_text = all_comment # train + test

cut_lines = jieba_lines(all_comment)

np.save("cut_lines", cut_lines)



cut_lines = np.load("cut_lines.npy")

cut_lines = cut_lines.tolist()

full_cut_text = cut_lines 
print("finish jieba")

embedding_model = Word2Vec(full_cut_text, size=300, window=4, min_count=1, iter=100)
#embedding_model = Word2Vec(full_cut_text, size=300, window=4, min_count=1,iter=50)
embedding_model.save("word2vec.model")
embedding_model = Word2Vec.load("word2vec.model")
existent_word = "姜太公"
if existent_word in embedding_model.wv.vocab:
    print(embedding_model.wv["姜太公"])
else:
    print("還我姜太公")
print("finish word2vec")


import pickle
tokenizer = Tokenizer(num_words=30000, filters="\n") #30000

tokenizer.fit_on_texts(full_cut_text)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

"""
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
"""
sequences = tokenizer.texts_to_sequences(full_cut_text)
train_sequences = tokenizer.texts_to_sequences(cut_lines)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

embedding_matrix = np.zeros((len(word_index)+1, 300))

for word, i in word_index.items():
    if word in embedding_model.wv.vocab:
        vector = embedding_model.wv[word]
        embedding_matrix[i] = vector
    else:
        print(word, " not in word2vec")
    
np.save("embedding_matrix", embedding_matrix)

embedding_matrix = np.load("embedding_matrix.npy")
print("embedd matrix shape: ",embedding_matrix.shape)


max_length = np.max([len(i) for i in sequences])
max_length = 100
print("max_length: ", max_length)
# without truncating
train_X_num = pad_sequences(train_sequences, maxlen=max_length)#truncating="post"
print("train_X_num: ", train_X_num.shape)
print(train_X_num[0])



trainx = train_X_num[:110000]
trainy = all_label[:110000]
validx = train_X_num[110000:]
validy = all_label[110000:]
print(trainx.shape, " ", trainy.shape)
print(validx.shape, " ", validy.shape)




bi_lstm = Bi_Lstm(max_length, word_index, 300, embedding_matrix)
model = bi_lstm.model

model.summary()
batch_size = 128
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
check_save = ModelCheckpoint("plot_modelW2V-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
model.fit(trainx, trainy, validation_data=(validx, validy),
            epochs=20, batch_size=batch_size,callbacks=[check_save])
exit(1)

