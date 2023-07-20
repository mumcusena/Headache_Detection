from keras.models import load_model
import numpy as np
from tensorflow import keras
from keras.models import Sequential

model = load_model('model.h5')
single_x_test = [41.0,1,1.0,8.0,3,1,0,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0
]
th = 0.5
q = model.predict(np.array( [single_x_test,] ))
print(type(q))
q[q[:, ] > th] = 1
q[q[:, ] < th] = 0
q[q[:, ] > th] = 1
q[q[:, ] < th] = 0
q[q[:, ] > th] = 1
q[q[:, ] < th] = 0
q[q[:, ] > th] = 1
q[q[:, ] < th] = 0
print(q)