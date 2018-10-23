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
    
    for i in ['LIMIT_BAL']:
        name = i+"square"
        numeric_attrs.append(name)
        data[name] = data[i]**2
    
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
    
    educateion = "EDUCATION" 
    edu = np.array(data[educateion].values)
    for i in range(edu.shape[0]):
        if edu[i] == 0 or edu[i] > 4:
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
    for i in cate_attrs[:3]:
        dummies_df = pd.get_dummies(data[i])
        
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)
    name = [-2,-1,0,1,2,3,4,5,6,7,8]
    for i in cate_attrs[3:]:
        dummies_df = pd.get_dummies(data[i])

        missing_cols = set( name ) - set( dummies_df.columns )
        for c in missing_cols:
            dummies_df[c] = 0
        dummies_df = dummies_df[name]
          
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)
    return data

def trans_pay_cate_attr(data, pay_cate_attr):
    print("trans_pay_cate_attr...")
    """
    pay_cate_attr_square=[]
    for i in pay_cate_attr:
        print(i)
        name = i+"square"
        pay_cate_attr_square.append(i)
        pay_cate_attr_square.append(name)
        
        data[name] = data[i]**2
    """
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
    """
    payment_attr_square = []
    for i in payment_attr_square:
        name = i+"square"
        payment_attr_square.append(i)
        payment_attr_square.append(name)
        
        data[name] = data[i]**2
    """
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
    """
    for i in cate_attrs:
        mean = data[i].mean()
        std = data[i].std()
        data[i] = (data[i] - mean) / std
    """
    #data = trans_pay_cate_attr(data, pay_cate_attr)
    data = trans_payment_attr(data, payment_attr)
    """
    col = ['BILL_AMT1', 'EDUCATION', 'SEX', 'PAY_AMT5', 'PAY_0', 'AGE', 'PAY_AMT2', 'PAY_AMT6', 'PAY_AMT3', 'PAY_AMT1', 'BILL_AMT6', 'PAY_AMT4', 'PAY_5', 'BILL_AMT3', 'BILL_AMT5']
    data = data[col]
    cate_attrs = ["SEX", "EDUCATION"]
    data = encode_cate_attrs(data, cate_attrs)
    """
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
    cate_attrs = ['SEX', 'EDUCATION','MARRIAGE','PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    data = fill_unknown(data, label_data, numeric_attrs, payment_attr, pay_cate_attr, cate_attrs)
    
    #data.to_csv(processed_data_path, index=False)

    return data, label_data


def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))

def class_mean_star(data):
    mean_star = np.mean(data, axis=0)
    return mean_star

data, label_data = preprocess_data()
print(data.shape)
column = data.columns

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

pc1 = len(pos_data)/(len(pos_data) + len(neg_data))
pc2 = len(neg_data)/(len(pos_data) + len(neg_data))
class_one_mean = class_mean_star(pos_data)
class_two_mean = class_mean_star(neg_data)
sigma_one = np.matmul(np.transpose((pos_data - class_one_mean)), (pos_data - class_one_mean)) / len(pos_data)
sigma_two = np.matmul(np.transpose((neg_data - class_two_mean)), (neg_data - class_two_mean)) / len(neg_data)
same_sigma = pc1 * sigma_one + pc2 * sigma_two


same_sigma_inv = np.linalg.pinv(same_sigma)
w = np.matmul((class_one_mean-class_two_mean), same_sigma_inv)

b = (-0.5)*np.dot(np.dot([class_one_mean], same_sigma_inv), class_one_mean) + (0.5)*np.dot(np.dot([class_two_mean], same_sigma_inv), class_two_mean)+np.log(len(pos_data)/len(neg_data))


total_xtrain = np.concatenate((pos_data, neg_data), axis=0)
total_ytrain = np.append(pos_label, neg_label)

ans = np.dot(total_xtrain, w) + b
ans = sigmoid(ans)


pridict = []
for i in ans:
    if i > 0.4:
        pridict.append(1)
    else:
        pridict.append(0)

acc = 0
for i in range(len(pridict)):
    if pridict[i] == total_ytrain[i]:
        acc += 1
print(acc/20000)    


np.save("gen_w", w)
np.save("gen_b", b)

num_nor = np.array(num_nor)
pay_cate_nor = np.array(pay_cate_nor)
pay_ment_nor = np.array(pay_ment_nor)

np.save("gen_num_nor", num_nor)
#np.save("pay_cate_nor1", pay_cate_nor)
np.save("gen_pay_ment_nor", pay_ment_nor)
