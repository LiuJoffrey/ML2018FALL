import numpy as np
import sys
import csv
import pandas as pd
from datetime import datetime
from numpy.linalg import inv
arg = sys.argv


def trans_num_attrs(data, numeric_attrs):
    num_nor = np.load("num_nor.npy")
    bining = [0,28,34,41,100] 
    bining_num = 4
    bining_attr = 'AGE'
    
    data[bining_attr] = pd.cut(data[bining_attr], bining)
    data[bining_attr] = pd.factorize(data[bining_attr])[0]
    
    print("trans_num_attrs...")
    count = 0
    for i in numeric_attrs:
        
        mean = num_nor[count,0]
        std = num_nor[count,1]
        data[i] = (data[i] - mean)/std
        #print(data[i])
        count += 1
        """
        max_ = data[i].max()
        min_ = data[i].min()
        data[i] = (data[i] - min_)/(max_-min_)
        #print(data[i])
        num_nor.append((max_, min_))
        """
    return data
def encode_cate_attrs(data, cate_attrs):
    
    educateion = "EDUCATION" 
    edu = np.array(data[educateion].values)
    for i in range(edu.shape[0]):
        if edu[i] > 4:
            edu[i] = 5
    data[educateion] = edu
    #print(data[educateion][130])
    """
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
    pay_cate_nor = np.load("pay_cate_nor.npy")
    count = 0
    for i in pay_cate_attr:
        
        mean = pay_cate_nor[count,0]
        std = pay_cate_nor[count, 1]
        data[i] = (data[i] - mean)/std
        #print(data[i])
        count += 1
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
    pay_ment_nor = np.load("pay_ment_nor.npy") 
    count = 0
    for i in payment_attr:
        
        mean = pay_ment_nor[count,0]
        std = pay_ment_nor[count,1]
        data[i] = (data[i] - mean)/std
        #print(data[i])
        count += 1
        """
        max_ = data[i].max()
        min_ = data[i].min()
        data[i] = (data[i] - min_)/(max_-min_)
        #print(data[i])
        pay_ment_nor.append((max_, min_))
        """
    return data

def fill_unknown(data, numeric_attrs, payment_attr, pay_cate_attr, cate_attrs):
    data = trans_num_attrs(data, numeric_attrs)
    data = encode_cate_attrs(data, cate_attrs)
    data = trans_pay_cate_attr(data, pay_cate_attr)
    data = trans_payment_attr(data, payment_attr)

    return data
def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

def preprocess_data():
    test_data_path = arg[3]
    print("Loading data...")
    data = pd.read_csv(test_data_path)
    
    numeric_attrs = ['LIMIT_BAL', 'AGE',]
    payment_attr = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5','BILL_AMT6', 
                    'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_cate_attr = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']   
    cate_attrs = ['SEX', 'EDUCATION','MARRIAGE']

    data = fill_unknown(data, numeric_attrs, payment_attr, pay_cate_attr, cate_attrs)
    
    return data




data = preprocess_data()
#data = np.concatenate((np.ones((data.shape[0],1)), data), axis=1)
#w = np.load("second.npy")

#hypo = np.dot(data,w)
#predict = sigmoid(hypo)
print(data.shape)
from numpy_neural_network import *
neural_network = NeuralNetwork()
neural_network.weigts_one = np.load("neural_network_weigts_one1.npy")
neural_network.bias_one = np.load("neural_network_bias_one1.npy")
neural_network.weighs_three = np.load("neural_network_weighs_three1.npy")
neural_network.bias_three = np.load("neural_network_bias_three1.npy")
predict = neural_network.predict(data)
#neural_network.accuracy(predicted, total_ytrain)
print(len(predict))


out = []
for i in range(len(predict)):
    if predict[i] >= 0.5:
        out.append(["id_"+str(i),1])
    else:
        out.append(["id_"+str(i),0])
submission = open(arg[4], "w+")
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(out)):
    s.writerow(out[i]) 
submission.close()
