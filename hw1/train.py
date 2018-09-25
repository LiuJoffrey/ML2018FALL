import numpy as np
import pandas as pd
import sys

arg = sys.argv

def GradientDescent(x,y,lr,tolerance=0.0001,iteration=100000):
    #theta = np.ones(x.shape[1])
    theta = np.random.rand(x.shape[1])
    #theta = np.zeros(x.shape[1])
    adagrad = np.zeros(x.shape[1])
    loss_history = []
    x_trans = x.transpose() * (-1)
    for i in range(iteration):
        var_ = np.dot(x, theta.transpose())
        diff = y - var_
        loss = np.sum(np.square(diff))/x.shape[0]
        loss_history.append(loss)
        gradient = 2 * np.dot(x_trans, diff) / x.shape[0]

        adagrad = adagrad+np.square(gradient)

        theta = theta - (lr/(np.sqrt(adagrad)+1))*gradient
        # theta = theta - lr*gradient
        # print("epoch: ", i+1, " loss: ", loss, " ", type(loss)," ",loss>tolerance)
        print("epoch: ", i+1, " loss: ", loss)
        if loss < tolerance:
            print("enough")
            return theta, loss_history
    return theta, loss_history

"""
### read csv data ###
data = pd.read_csv(arg[1], encoding = "ISO-8859-1")
data = data.iloc[:, 3:]
day_data = []
day_count = data.shape[0] // 18

### get each day data ###
for i in range(day_count):
    day = data.iloc[i*18:(i+1)*18]
    day = day.reset_index(drop=True)
    day_data.append(day)

data = pd.concat(day_data, axis=1, ignore_index=True)

### convert RAINFALL NR to zero ###

for i in range(data.shape[1]):
    if data.iloc[10,i] == 'NR':
        data.iloc[10,i] = 0
column_name = range(data.shape[1])
data.columns = column_name
print(data.shape)
preprocessingfile = np.array(data.values)

np.save("propress_18_5760", preprocessingfile)
"""
data = np.load("propress_18_5760.npy")
normal_min_max = []

for i in range(data.shape[0]):
    feature = data[i].astype(float)
    
    feature_min = min(feature)
    feature_max = max(feature)
    nor_feature = (feature-feature_min)/(feature_max-feature_min)
    data[i] = nor_feature
    normal_min_max.append((feature_min, feature_max))


normal_min_max = np.array(normal_min_max)
np.save("normal_min_max", normal_min_max)
print(normal_min_max.shape)


x_data = []
y_data = []
total_column = data.shape[1]
i = 0
data_count = 0
while i < (total_column-9):
    first_hour = i
    last_hour = i+10
    ten_hour = data[:,last_hour-1]
    ten_hour = ten_hour.astype(float)
    nine_hour = data[:,first_hour:last_hour-1]
    nine_hour = nine_hour.astype(float)
    nine_hour_flat = nine_hour.flatten()
    nine_hour_flat = np.append(nine_hour_flat, [1])
    ylabel = ten_hour[9]
    #print(nine_hour_flat.shape)
    #print(nine_hour_flat)
    x_data.append(nine_hour_flat)
    y_data.append(ylabel)
    i+=1
    data_count += 1
    if data_count > 471:
        i = i-1+10
        data_count = 0
    
    #print(nine_hour.values)
    #print(ylabel)
#print(data.iloc[9,:])


x_data = np.array(x_data)
x_data.astype(float)
print(x_data.shape)
y_data = np.array(y_data)
y_data.astype(float)
print(y_data.shape)


#np.save("x_data_raw_162", x_data)
#np.save("y_data_raw", y_data)



lr = np.array([1]*x_data.shape[1])
#lr = 0.000001
weight, loss_history = GradientDescent(x_data,y_data,lr)
print(weight)
np.save("162_feature_weight", weight)

# data.to_csv(arg[2])


weight = np.load("162_feature_weight.npy")
print(weight.shape)
print(x_data.shape)
var_ = np.dot(x_data, weight.transpose())
print(var_.shape)
print(normal_min_max[9])
var_ = var_ * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]
print(var_.shape)
diff = np.absolute(y_data - var_)
print("max: ", max(diff), " min: ", min(diff))
diff_err = np.sum(np.absolute(y_data - var_)) / var_.shape[0]
print(diff_err)
