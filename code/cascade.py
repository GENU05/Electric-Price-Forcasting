import pandas as pd 
df = pd.read_csv('Datasets/newcreate.csv')
# print(df.head(5))
# df = df.drop(['Date','DA_CC','RT_CC','DewPnt','SYSLoad'],axis=1)
price_back = (df.max()-df.min())
value_array = price_back.values 
price = value_array[10]
#Normalization of data
df = df/(df.max()-df.min())
# print(df.head(5))

def MSE(y,y_pred):
    #y is a list
    ans = 0
    for j in range(len(y)):
        ans += abs(y[j] - y_pred[j][0])**2
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
        # print(y[j]*price,'::::',y_pred[j][0]*price)
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

from keras import initializers

X=df.drop(['RegCP'],axis=1)
Y=df.iloc[:,10] 
# print(X.head(10))
# print(Y.head(5))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 

def build_model():
    dense_input = Input(shape=(10,))
    dense_vector = BatchNormalization()(dense_input)
    print(dense_vector)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector)
    Dense(7,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector)
    Dense(8,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector)
    # Dense(17)(dense_vector)

    output = Dense(1)(dense_vector)
    model = Model(inputs=[dense_input], outputs=output)
    return model

model = build_model()
model.compile(loss="mean_absolute_percentage_error", optimizer="sgd", metrics=['accuracy'])
model.fit([X_train.values], Y_train.values, epochs=10, batch_size=10)
y_pred = model.predict([X_test]) 
y = Y_test.tolist() 

RR(y,y_pred)
MAPE(y,y_pred)
MSE(y,y_pred)
y = np.multiply(y,price)
y_pred = np.multiply(y_pred,price)
import matplotlib.pyplot as plt
plt.plot(y[:31],color='red')
plt.plot(y_pred[:31])
# plt.axis([0, 6, 0, 20])
plt.show()
    


# ----Ploting----
import matplotlib.pyplot as plt
plt.plot(y,color='red')
plt.plot(y_pred)
# plt.axis([0, 6, 0, 20])
plt.show()
