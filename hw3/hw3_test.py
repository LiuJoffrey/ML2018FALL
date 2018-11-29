
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import sys
from model import *
arg = sys.argv

raw = pd.read_csv(arg[1])
data = raw.iloc[:,1]
data = data.str.split(expand = True).astype('float32').values
data = data.reshape(-1,48,48,1)

data = data/255

# model1_00218_0.71013.h5 0.69713
# modelAtten_3_Alexnet_00342_0.71621.h5 0.70325
model1 = load_model('modelAtten_3_Alexnet_00342_0.71621.h5')
model2 = load_model("model_reloadAlexnet_0.72588l.h5")
model3 = load_model("1_modelAlexnet_00079_0.80600.h5")

#model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y1 = model1.predict(data)
y2 = model2.predict(data)
y3 = model3.predict(data)
y = (y1+y2+y3)/3


ans = np.argmax(y,axis=1)

with open(arg[2],'w') as output:
    output.write('id,label\n')
    for i in range(y.shape[0]):
        output.write(str(i)+','+str(ans[i])+'\n')



"""
alexnet = Atten_custom_3_AlexNet(data.shape[1:])
model = alexnet.model
model.load_weights("modelAtten_3_Alexnet_00349_0.71445.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""