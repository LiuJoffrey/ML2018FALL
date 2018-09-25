import numpy as np
import pandas as pd
import sys
import csv

arg = sys.argv
data = pd.read_csv(arg[1], encoding = "ISO-8859-1", header=None)
day_count = data.shape[0] // 18

x_data = []
y_data = []
normal_min_max = np.load("normal_min_max.npy")
weight = np.load("162_feature_weight.npy")
for i in range(day_count):
    day = data.iloc[i*18:(i+1)*18,2:]
    day = np.array(day.values)
    for j in range(day.shape[1]):
        if day[10, j] == "NR":
            day[10, j] = 0

    
    day = day.astype(float)
    for j in range(day.shape[0]):
        feature = day[j]
        #print(j,": ",feature.shape)
        nor_feature = (feature-normal_min_max[j][0])/(normal_min_max[j][1]-normal_min_max[j][0])
        day[j] = nor_feature
    day = day.flatten()
    day = np.append(day, [1])
    
    x_data.append(day)

x_data = np.array(x_data)
print(x_data.shape)

var_ = np.dot(x_data, weight.transpose())
var_ = var_ * (normal_min_max[9][1]-normal_min_max[9][0]) + normal_min_max[9][0]

print(var_.shape)

id_ = np.array(["id"])
value = np.array(["value"])
id_name = []
out = []
for i in range(var_.shape[0]):
    if var_[i] < 0:
        var_[i] = 0
    out.append(["id_"+str(i),var_[i]])
    
"""
out = np.array(out)
print(out.shape)
out = pd.DataFrame(out,columns=['id', 'value'])
print(out)
out.to_csv(arg[2])
#print(id_)
#print(var_)
"""
submission = open(arg[2], "w+")
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(out)):
    s.writerow(out[i]) 
submission.close()