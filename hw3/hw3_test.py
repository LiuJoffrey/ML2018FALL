
from keras.models import Model, load_model
import numpy as np
import pandas as pd
import sys
import cv2

arg = sys.argv

raw = pd.read_csv(arg[1])
data = raw.iloc[:,1]
data = data.str.split(expand = True).astype('float32').values
data = data.reshape(-1,48,48,1)

data = data/255

model = load_model('model1_00114_0.65831.h5')

y = model.predict(data)
ans = np.argmax(y,axis=1)

with open(arg[2],'w') as output:
    output.write('id,label\n')
    for i in range(y.shape[0]):
        output.write(str(i)+','+str(ans[i])+'\n')