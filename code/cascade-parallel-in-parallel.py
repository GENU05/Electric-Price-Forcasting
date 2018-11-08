import pandas as pd 
df = pd.read_csv('Datasets/newcreate.csv')
# print(df.head(5))
# df = df.drop(['Date'],axis=1)
# print(df.head(5))

def MSE(y,y_pred):
    #y is a list
    ans = 0
    for j in range(len(y)):
        ans += (y[j] - y_pred[j][0])**2
        # ans = ans/len(X_test)
    ans = ( ans / len(y)**2 )
    print('MSE:',ans)
    print('Length of Y',len(y))
    print('-'*10)
    return ans

def MAPE(y,y_pred):
    a = 0
    for j in range(len(y)):
        a += abs((y[j] - y_pred[j][0])/y[j])
    a = a*100/len(y)
    print('MAPE:',a)
    return a 

def RR(y,y_pred):
    y_avg=0
    p = 0
    q=0
    for i in range(len(y)):
        y_avg += y[i]
    for i in range(len(y)):
        p += (y_pred[i][0]-y_avg)
        q += (y[i]-y_avg)
    print('R^2:',p/q)
    return p/q

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
from keras.models import Sequential , Model 
from keras.layers import *

from keras.layers.normalization import BatchNormalization
np.random.seed(7)


X=df.drop(['RegCP'],axis=1)
Y=df.iloc[:,10] 
# print(X.head(10))
# print(Y.head(5))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1000) 



def build_model():
    #---cascade----
    dense_input = Input(shape=(10,))
    dense_vector = BatchNormalization()(dense_input)
    print(dense_vector)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector)
    Dense(7)(dense_vector)
    Dense(8)(dense_vector)
    Dense(1)(dense_vector)
    #---parallel---
    dense_input3 = Input(shape=(10,))
    dense_vector3 = BatchNormalization()(dense_input3)
    Dense(6)(dense_vector3)

    dense_input1 = Input(shape=(10,))
    dense_vector1 = BatchNormalization()(dense_input1)
    Dense(7)(dense_vector1)

    dense_input2 = Input(shape=(10,))
    dense_vector2 = BatchNormalization()(dense_input2)
    Dense(8)(dense_vector2)

    feature_vector2 = concatenate([dense_vector3, dense_vector1 , dense_vector2]) #merge parallel
    #--merge cascade and parallel--
    feature_vector = concatenate([feature_vector2,dense_vector])
    #--output--
    output = Dense(1)(feature_vector)
    model = Model(inputs=[dense_input,dense_input3,dense_input1,dense_input2], outputs=output)
    return model


model = build_model()
model.compile(loss="mean_absolute_percentage_error", optimizer="adam", metrics=['accuracy'])
model.fit([X_train.values]*4, Y_train.values, epochs=10, batch_size=10)
y_pred = model.predict([X_test]*4) 
y = Y_test.tolist() 

RR(y,y_pred)
MAPE(y,y_pred)
MSE(y,y_pred)
    
# ----Ploting----
import matplotlib.pyplot as plt
plt.plot(y,color='red')
plt.plot(y_pred)
# plt.axis([0, 6, 0, 20])
plt.show()


