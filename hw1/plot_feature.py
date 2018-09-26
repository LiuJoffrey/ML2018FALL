import matplotlib.pyplot as plt
import numpy as np
import math
data = np.load("propress_18_5760.npy")


for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data[i,j] = float(data[i,j])
pm2 = data[9]
feature_name = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR',
'WIND_DIREC','WIND_SPEED','WS_HR']


for i in range(data.shape[0]):
    feature = data[i]
    feature_min = min(feature)
    feature_max = max(feature)
    nor_feature = (feature-feature_min)/(feature_max-feature_min)
    data[i] = nor_feature
    


for i in range(data.shape[0]):
    x = data[i]
    y = pm2
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    molecular = np.dot((x-x_mean),(y-y_mean)) #

    x_diff = x - x_mean
    y_diff = y - y_mean
    x_diff_squ = x_diff**2
    y_diff_squ = y_diff**2
    Denominator = np.sqrt(np.sum(x_diff_squ)) * np.sqrt(np.sum(y_diff_squ))

    cor = molecular/Denominator
    #print(molecular/Denominator)
    if math.fabs(cor) > 0.066:
        print("Index: ",i ," name: ", feature_name[i] , " cor: ", cor) 
#print(np.corrcoef(data[0],data[9]))


"""

name:  AMB_TEMP  cor:  -0.06601482639716412
name:  CH4  cor:  0.13388015767749056
name:  CO  cor:  0.2934257336962047
name:  NMHC  cor:  0.23216153258615133
name:  NO  cor:  0.1108721124106322
name:  NO2  cor:  0.27480806751376485
name:  NOx  cor:  0.2458166789579479
name:  O3  cor:  0.06845256337280901
name:  PM10  cor:  0.4947344360188051
name:  PM2.5  cor:  0.9999999999999999
name:  RAINFALL  cor:  -0.03413822894113984
name:  RH  cor:  -0.0062311826076463005
name:  SO2  cor:  0.24643195907666104
name:  THC  cor:  0.213072774164363
name:  WD_HR  cor:  0.0551366667167957
name:  WIND_DIREC  cor:  0.05499824832295328
name:  WIND_SPEED  cor:  -0.06154939143420278
name:  WS_HR  cor:  -0.04658693613780084
"""