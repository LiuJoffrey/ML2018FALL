import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
from itertools import combinations
arg = sys.argv

w = np.load('11_6_feature_weight.npy')
normal_min_max = np.load('normal_min_max.npy')
test_x = []
n_row = 0
text = open(arg[1] ,"r")
row = csv.reader(text , delimiter= ",")
feature = [4,5,6,7,8,9]
for r in row:
    if n_row %18 == 0:
        test_x.append([])
        if n_row in feature:
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]))
    else :
        if (n_row%18) in feature:
            for i in range(2,11):
                if r[i] !="NR":
                    test_x[n_row//18].append((float(r[i])-normal_min_max[(n_row%18)][1])/(normal_min_max[(n_row%18)][0]-normal_min_max[(n_row%18)][1]))
                else:
                    test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)
print(test_x.shape)
# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((test_x,np.ones((test_x.shape[0],1))), axis=1)

print(test_x.shape)

test_after = []
for i in range(test_x.shape[0]):
    each_after = []
    for j in range(6):
        d = test_x[i, j*9:(j+1)*9]
        for k in range(len(d)):
            if k == 0:
                if d[k]<=0:
                    d[k] = d[k+1]
            if k == len(d)-1:
                if d[k]<=0:
                    d[k] = d[k-1]
            else:
                if d[k]<=0:
                    d[k] = (d[k-1]+d[k+1])/2
            each_after.append(d[k])
    test_after.append(each_after)
test_after = np.array(test_after)
test_after = np.concatenate((test_after,np.ones((test_after.shape[0],1))), axis=1)
print(test_after.shape)
print(test_after)
var_ = np.dot(test_after, w)
var_ = var_ * (normal_min_max[9][0]-normal_min_max[9][1]) + normal_min_max[9][1]
print(var_.shape)
print(var_)

id_ = np.array(["id"])
value = np.array(["value"])
id_name = []
out = []
for i in range(var_.shape[0]):
    if var_[i] < 0:
        var_[i] = 0
    out.append(["id_"+str(i),var_[i]])
  

submission = open(arg[2], "w+")
s = csv.writer(submission,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(out)):
    s.writerow(out[i]) 
submission.close()



