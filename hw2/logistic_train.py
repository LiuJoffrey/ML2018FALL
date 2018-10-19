import numpy as np
import sys
import csv
import pandas as pd
from datetime import datetime
from numpy.linalg import inv

arg = sys.argv

column = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



age_bin = [0,28,34,41,100]
num_nor = []
#cate_nor = []
pay_cate_nor = []
pay_ment_nor = []
def trans_num_attrs(data, numeric_attrs):
    # [<25], [25~30], [35~40], [40~45], [>45]
    bining = [0,28,34,41,100] 
    bining_num = 4
    bining_attr = 'AGE'
    
    #data[bining_attr] = pd.qcut(data[bining_attr], bining_num)
    data[bining_attr] = pd.cut(data[bining_attr], bining)
    #bining = pd.factorize(data[bining_attr])[1]
    # [(20.999, 28.0], (34.0, 41.0], (41.0, 79.0], (28.0, 34.0]]
    
    data[bining_attr] = pd.factorize(data[bining_attr])[0]
    
    #data[bining_attr] = pd.factorize(data[bining_attr])[0]+1
    print("trans_num_attrs...")
    
    for i in numeric_attrs:
        
        mean = data[i].mean()
        std = data[i].std()
        data[i] = (data[i] - mean)/std
        #print(data[i])
        num_nor.append((mean, std))
        """
        max_ = data[i].max()
        min_ = data[i].min()
        data[i] = (data[i] - min_)/(max_-min_)
        #print(data[i])
        num_nor.append((max_, min_))
        """
    return data

def encode_cate_attrs(data, cate_attrs):
    """
    educateion = "EDUCATION" 
    edu = np.array(data[educateion].values)
    for i in range(edu.shape[0]):
        if edu[i] == 0 or edu[i] > 4:
            edu[i] = 4
    data[educateion] = edu
    #print(data[educateion][130])

    marrige = 'MARRIAGE'
    mar = np.array(data[marrige].values)
    for i in range(mar.shape[0]):
        if mar[i] == 0:
            mar[i] = 3
    data[marrige] = mar
    """
    print("encode_cate_attrs...")
    for i in cate_attrs[:]:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)

    return data

def trans_pay_cate_attr(data, pay_cate_attr):
    print("trans_pay_cate_attr...")
    for i in pay_cate_attr:
        
        mean = data[i].mean()
        std = data[i].std()
        data[i] = (data[i] - mean)/std
        #print(data[i])
        pay_cate_nor.append((mean, std))
        """
        max_ = data[i].max()
        min_ = data[i].min()
        data[i] = (data[i] - min_)/(max_-min_)
        #print(data[i])
        pay_cate_nor.append((max_, min_))
        """
    return data

def trans_payment_attr(data, payment_attr):
    print("trans_payment_attr...")
        
    for i in payment_attr:
        
        mean = data[i].mean()
        std = data[i].std()
        data[i] = (data[i] - mean)/std
        #print(data[i])
        pay_ment_nor.append((mean, std))
        """
        max_ = data[i].max()
        min_ = data[i].min()
        data[i] = (data[i] - min_)/(max_-min_)
        #print(data[i])
        pay_ment_nor.append((max_, min_))
        """
    return data

def fill_unknown(data, label_data, numeric_attrs, payment_attr, pay_cate_attr, cate_attrs):
    data = trans_num_attrs(data, numeric_attrs)
    data = encode_cate_attrs(data, cate_attrs)
    data = trans_pay_cate_attr(data, pay_cate_attr)
    data = trans_payment_attr(data, payment_attr)

    return data

def preprocess_data():
    input_data_path = arg[1]
    input_data_label_path = arg[2]

    #processed_data_path = 'all/processed_data.csv'
    print("Loading data...")

    data = pd.read_csv(input_data_path)
    label_data = pd.read_csv(input_data_label_path)
    
    numeric_attrs = ['LIMIT_BAL', 'AGE',]
    payment_attr = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5','BILL_AMT6', 
                    'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_cate_attr = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']   
    cate_attrs = ['SEX', 'EDUCATION','MARRIAGE']

    data = fill_unknown(data, label_data, numeric_attrs, payment_attr, pay_cate_attr, cate_attrs)
    
    #data.to_csv(processed_data_path, index=False)

    return data, label_data


def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))


data, label_data = preprocess_data()
### positive and negitive ###
positive_data = data[label_data["Y"]==1] # 4445
positive_label = label_data[label_data["Y"]==1]
negtive_data = data[label_data["Y"]==0] # 15555
negtive_label = label_data[label_data["Y"]==0]

pos_data = np.array(positive_data.values)
pos_label = np.array(positive_label.values)
pos_label = np.reshape(pos_label, (pos_label.shape[0],))
neg_data = np.array(negtive_data.values)
neg_label = np.array(negtive_label.values)
neg_label = np.reshape(neg_label, (neg_label.shape[0],))

pos_data_train = pos_data[:4000]
pos_label_train = pos_label[:4000]
pos_data_val = pos_data[4000:]
pos_label_val = pos_label[4000:]
neg_data_train = neg_data[:14000]
neg_label_train = neg_label[:14000]
neg_data_val = neg_data[14000:]
neg_label_val = neg_label[14000:]
 
val_data = np.concatenate((pos_data_val, neg_data_val), axis=0)
val_label = np.append(pos_label_val, neg_label_val)

#total_xtrain = np.concatenate((pos_data_train, neg_data_train), axis=0)
#total_ytrain = np.append(pos_label_train, neg_label_train)

total_xtrain = np.concatenate((pos_data, neg_data), axis=0)
total_ytrain = np.append(pos_label, neg_label)

randomize = np.arange(len(total_xtrain))
np.random.shuffle(randomize)
total_xtrain = total_xtrain[randomize]
total_ytrain = total_ytrain[randomize]


print(val_data.shape, " ,", val_label.shape)
print(total_xtrain.shape, " ,", total_ytrain.shape)

from numpy_neural_network import *

neural_network = NeuralNetwork(epochs=40000)
neural_network.train(total_xtrain, total_ytrain)
predicted = neural_network.predict(total_xtrain)
neural_network.accuracy(predicted, total_ytrain)

print(neural_network.weigts_one.shape, ", ", neural_network.bias_one.shape)
print(neural_network.weighs_three.shape, ", ", neural_network.bias_three.shape)
print(len(predicted))

np.save("neural_network_weigts_one", neural_network.weigts_one)
np.save("neural_network_bias_one", neural_network.bias_one)
np.save("neural_network_weighs_three", neural_network.weighs_three)
np.save("neural_network_bias_three", neural_network.bias_three)

age_bin = [0,28,34,41,100]
num_nor = np.array(num_nor)
pay_cate_nor = np.array(pay_cate_nor)
pay_ment_nor = np.array(pay_ment_nor)

np.save("num_nor", num_nor)
np.save("pay_cate_nor", pay_cate_nor)
np.save("pay_ment_nor", pay_ment_nor)