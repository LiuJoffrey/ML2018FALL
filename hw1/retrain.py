import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
from itertools import combinations
arg = sys.argv



def gradient_descent(x,y):
    w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
    """
    w = np.zeros(len(x[0]))
    l_rate = 10
    repeat = 10000
    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))

    for i in range(repeat):
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        gra = np.dot(x_t,loss)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra/ada
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
    
    #w = np.dot(np.dot(inv(np.dot(x,x.transpose())), x), y)
    #w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
    """
    return w




data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open(arg[1], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                if float(r[i]) < 0:
                    r[i] = (-1) * float(r[i])
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()
data = np.array(data)

square_feature = [9]
for i in square_feature:
    f = data[i]
    square_f = f**2
    square_f = square_f.reshape((1,5760))
    data = np.concatenate((data,square_f),axis=0)

####  data include pm2.5**2, total 19 rows ####

normal_min_max = []

for i in range(data.shape[0]):
    feature = data[i]
    max_ = max(feature)
    min_ = min(feature)
    feature = (feature-min_)/(max_-min_)
    normal_min_max.append((max_,min_))
    data[i] = feature
normal_min_max = np.array(normal_min_max)
np.save("normal_min_max",normal_min_max)

print(data.shape)
y_label = data[9]
total_try = []
for i in range(1,20):
    index = np.arange(0,19)
    c = list(combinations(index, i))
    for row in c:
        row = list(row)
        #print(row)
        total_try.append(row)
print(len(total_try))
total_try = []
total_try.append([2,4,5,6,7,8,9,12])
min_error = 100
combination = []
min_error_12 = []
total_good_combin = []
total_try = total_try[::-1]
for try_index in range(len(total_try))[:]:
    print("Try: ", try_index)
    feature = total_try[try_index]
    
    x_data = []
    y_data = []
    # 12 month
    for i in range(12):
        # 一個月連續10小時的data有471筆
        for j in range(471):
            x_data.append([])
            # 19 污染物
            for t in range(19):
                # 嘗試各種組合
                if t in feature:
                    for s in range(9):
                        x_data[471*i+j].append(float(data[t][480*i+j+s]))

            y_data.append(float(y_label[480*i+j+9]))
    

    x_data = np.array(x_data)
    x_data = np.concatenate((x_data,np.ones((x_data.shape[0],1))), axis=1)
    
    ### delete 3month in x_data ###
    first_part = x_data[:7*157]
    second_part = x_data[8*157:9*157]
    last_part = x_data[10*157:]
    x_data = np.append(first_part,second_part, axis=0)
    x_data = np.append(x_data, last_part,axis=0)
    
    y_data = np.array(y_data)
    
    first_part = y_data[:7*157]
    second_part = y_data[8*157:9*157]
    last_part = y_data[10*157:]
    y_data = np.append(first_part,second_part, axis=0)
    y_data = np.append(y_data,last_part, axis=0)
    
    total_x_data = x_data
    total_y_data = y_data
    average_error = 0
    error_12 = []
    
    # 3 4 月 不太好
    # 12 month validation, 11 is 12 months - 4 moths
    # 36 is 157 for a batch
    for vali_index in range(34):
        if vali_index == 0:
            vali = x_data[vali_index*157:vali_index*157+157]
            vali_label = y_data[vali_index*157:vali_index*157+157]
            train = x_data[(vali_index+1)*157:]
            train_label = y_data[(vali_index+1)*157:]
            
        elif vali_index == 35:
            vali = x_data[vali_index*157:vali_index*157+157]
            train = x_data[:(vali_index)*157]
            vali_label = y_data[vali_index*157:vali_index*157+157]
            train_label = y_data[:(vali_index)*157]
        else:
            
            vali = x_data[vali_index*157:vali_index*157+157]
            vali_label = y_data[vali_index*157:vali_index*157+157]
            first = x_data[:vali_index*157]
            first_label = y_data[:vali_index*157]
            last = x_data[(vali_index+1)*157:]
            last_label = y_data[(vali_index+1)*157:]
            train = np.append(first,last, axis=0)
            train_label = np.append(first_label,last_label, axis=0)
    
        """
        train_after = []
        train_label_after = []
        
        for i in range(train.shape[0]):
            if_zero = False
            for j in range(train.shape[1])[45:]:
                if train[i,j] <= 0:
                    if_zero = True
                    continue
            if if_zero is not True:
                train_after.append(train[i])
                train_label_after.append(train_label[i])
        
        train = np.array(train_after)
        train_label = np.array(train_label_after)
        print("train_after: ", train.shape)
        print("train_label_after: ", train_label.shape)
        """
        """
        vali_after = []
        vali_label_after = []
        
        for i in range(vali.shape[0]):
            for j in range(vali.shape[1]):
                if j % 9
        """

        weight = gradient_descent(train,train_label)
        var_ = np.dot(vali, weight)
        var_ = var_ * (normal_min_max[9][0]-normal_min_max[9][1]) + normal_min_max[9][1]
        vali_label = vali_label * (normal_min_max[9][0]-normal_min_max[9][1]) + normal_min_max[9][1]
        diff = vali_label - var_
        cost = np.sum(diff**2) / len(var_)
        cost_a  = math.sqrt(cost)
        average_error += cost_a
        error_12.append(cost_a)



    average_error = average_error/34
    print("com: ", feature)
    print("average_error: ", average_error)
    print("error_12: ", error_12)
    if average_error < min_error:
        min_error = average_error
        combination = feature
        min_error_12 = error_12
        np.save("best_try_feature_weight", weight)
    if average_error < 22.1:
        total_good_combin.append(feature)
        
    print("min_error: ", min_error)
    print("min_combination: ", combination)
    print("min_error_12: ", min_error_12)
    print("len(total_good_combin): ", len(total_good_combin))
    #print("total_good_combin: ", total_good_combin)
    print()


    weight = gradient_descent(total_x_data,total_y_data)
    var_ = np.dot(total_x_data, weight)
    var_ = var_ * (normal_min_max[9][0]-normal_min_max[9][1]) + normal_min_max[9][1]
    total_y_data = total_y_data * (normal_min_max[9][0]-normal_min_max[9][1]) + normal_min_max[9][1]
    diff = total_y_data - var_
    cost = np.sum(diff**2) / len(var_)
    cost_a  = math.sqrt(cost)

    print("final: ", cost_a)
    print()
    print(weight)
    np.save("after_try", weight)

"""
    for vali_index in range(12):
        if vali_index == 0:
            vali = x_data[vali_index*471:vali_index*471+471]
            vali_label = y_data[vali_index*471:vali_index*471+471]
            train = x_data[(vali_index+1)*471:]
            train_label = y_data[(vali_index+1)*471:]
            
        elif vali_index == 11:
            vali = x_data[vali_index*471:vali_index*471+471]
            train = x_data[:(vali_index)*471]
            vali_label = y_data[vali_index*471:vali_index*471+471]
            train_label = y_data[:(vali_index)*471]
        else:
            
            vali = x_data[vali_index*471:vali_index*471+471]
            vali_label = y_data[vali_index*471:vali_index*471+471]
            first = x_data[:vali_index*471]
            first_label = y_data[:vali_index*471]
            last = x_data[(vali_index+1)*471:]
            last_label = y_data[(vali_index+1)*471:]
            train = np.append(first,last, axis=0)
            train_label = np.append(first_label,last_label, axis=0)
"""


"""
    weight = gradient_descent(x_data,y_data)
    var_ = np.dot(x_data, weight)
    var_ = var_ * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]
    y_data = y_data * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]
    diff = y_data - var_
    cost = np.sum(diff**2) / len(var_)
    cost_a  = math.sqrt(cost)
    print("com: ", feature)
    print("Cost: ", cost_a)
    if cost_a < min_error:
        min_error = cost_a
        combination = feature

    print("min_error: ", min_error)
    print()
    if cost_a < 22.1:
        total_good_combin.append(feature)
print("min_error: ", min_error)
print("combination: ", combination)
print("len(total_good_combin): ", len(total_good_combin))
print("total_good_combin: ", total_good_combin)
"""

"""
x_data = []
y_data = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x_data.append([])
        # 18種污染物+pm2.5**2
        for t in range(19):
            # 連續9小時
            for s in range(9):
                x_data[471*i+j].append(float(data[t][480*i+j+s]) )
        
        y_data.append(float(y_label[480*i+j+9]))


x_data = np.array(x_data)
x_data = np.concatenate((x_data,np.ones((x_data.shape[0],1))), axis=1)
y_data = np.array(y_data)

print(x_data.shape)

weight = gradient_descent(x_data,y_data)

var_ = np.dot(x_data, weight)

var_ = var_ * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]
y_data = y_data * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]
diff = y_data - var_
cost = np.sum(diff**2) / len(var_)
cost_a  = math.sqrt(cost)
print(cost_a)
"""



