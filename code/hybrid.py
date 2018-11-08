import pandas as pd 
df = pd.read_csv('Datasets/newcreate.csv')
# print(df.head(5))
# df = df.drop(['Date'],axis=1)
# print(df.head(5))

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
    #***1st Layer***
    #---cascade----
    dense_input1 = Input(shape=(10,))
    dense_vector1 = BatchNormalization()(dense_input1)
    print(dense_vector1)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector1)
    Dense(7)(dense_vector1)
    Dense(8)(dense_vector1)
    op = Dense(1)(dense_vector1)
    #---parallel---
    # [dense_input1] : 1
    dense_input13 =op # Input(shape=(1,))
    dense_vector13 = BatchNormalization()(dense_input13)
    Dense(6)(dense_vector13)

    dense_input11 = op#Input(shape=(1,))
    dense_vector11 = BatchNormalization()(dense_input11)
    Dense(7)(dense_vector11)

    dense_input12 = op#Input(shape=(1,))
    dense_vector12 = BatchNormalization()(dense_input12)
    Dense(8)(dense_vector12)

    feature_vector12 = concatenate([dense_vector13, dense_vector11 , dense_vector12]) #merge parallel
    #--merge cascade and parallel--
    feature_vector1 = concatenate([feature_vector12,dense_vector1])
    #--output--
    output1 = Dense(1)(feature_vector1) 

    #***2nd layer****

    #---parallel---
    dense_input23 = Input(shape=(10,))
    dense_vector23 = BatchNormalization()(dense_input23)
    Dense(6)(dense_vector23)

    dense_input21 = Input(shape=(10,))
    dense_vector21 = BatchNormalization()(dense_input1)
    Dense(7)(dense_vector21)

    dense_input22 = Input(shape=(10,))
    dense_vector22 = BatchNormalization()(dense_input22)
    Dense(8)(dense_vector22)

    #[dense_input1,dense_input23,dense_input21,dense_input22] : 4 
    feature_vector22 = concatenate([dense_vector23, dense_vector21 , dense_vector22]) #merge parallel
    op = Dense(1)(feature_vector22)
    #---cascade----
    dense_input2 = op
    dense_vector2 = BatchNormalization()(dense_input2)
    print(dense_vector2)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector2)
    Dense(7)(dense_vector2)
    Dense(8)(dense_vector2)
    output2 = Dense(1)(dense_vector2)
    #--merge cascade and parallel--
    # feature_vector = concatenate([feature_vector2,dense_vector])
    #--output--
    # output = Dense(1)(feature_vector)

    #***3rd Layer***
    dense_input3 = Input(shape=(10,))
    dense_vector3 = BatchNormalization()(dense_input3)
    print(dense_vector3)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector3)
    Dense(7)(dense_vector3)
    Dense(8)(dense_vector3)
    Dense(1)(dense_vector3)
    #---parallel---
    #[dense_input1,dense_input23,dense_input21,dense_input22,dense_input3,dense_input33,dense_input31,dense_input32] : 8
    dense_input33 = Input(shape=(10,))
    dense_vector33 = BatchNormalization()(dense_input33)
    Dense(6)(dense_vector33)

    dense_input31 = Input(shape=(10,))
    dense_vector31 = BatchNormalization()(dense_input31)
    Dense(7)(dense_vector31)

    dense_input32 = Input(shape=(10,))
    dense_vector32 = BatchNormalization()(dense_input32)
    Dense(8)(dense_vector32)

    feature_vector32 = concatenate([dense_vector33, dense_vector31 , dense_vector32]) #merge parallel
    #--merge cascade and parallel--
    feature_vector3 = concatenate([feature_vector32,dense_vector3])
    #--output--
    output3 = Dense(1)(feature_vector3)

    #***4th LAyer***
    dense_input4 = Input(shape=(10,))
    dense_vector4 = BatchNormalization()(dense_input4)
    print(dense_vector4)
    Dense(6,kernel_initializer=initializers.RandomNormal(stddev=0.001))(dense_vector4)
    Dense(7)(dense_vector4)
    Dense(8)(dense_vector4)
    Dense(1)(dense_vector4)
    #---parallel---
    dense_input43 = Input(shape=(10,))
    dense_vector43 = BatchNormalization()(dense_input43)
    Dense(6)(dense_vector43)

    dense_input41 = Input(shape=(10,))
    dense_vector41 = BatchNormalization()(dense_input41)
    Dense(7)(dense_vector41)

    dense_input42 = Input(shape=(10,))
    dense_vector42 = BatchNormalization()(dense_input42)
    Dense(8)(dense_vector42)
    #[dense_input1,dense_input23,dense_input21,dense_input22,dense_input3,dense_input33,dense_input31,dense_input32,dense_input4,dense_input43,dense_input41,dense_input42] :12
    feature_vector42 = concatenate([dense_vector43, dense_vector41 , dense_vector42]) #merge parallel
    #--merge cascade and parallel--
    feature_vector4 = concatenate([feature_vector42,dense_vector4])
    #--output--
    output4 = Dense(1)(feature_vector4) 

    #***FInal Merge*** 
    # final_input = list()
    # for i in range(len(output1)):
    #     final_input.append(output1[i]+output2[i]+output3[i]+output4[i])
    final = concatenate([feature_vector4,feature_vector3,feature_vector42,feature_vector1]) 
    output = Dense(1)(final) 



    model = Model(inputs=[dense_input1,dense_input23,dense_input21,dense_input22,dense_input3,dense_input33,dense_input31,dense_input32,dense_input4,dense_input43,dense_input41,dense_input42], outputs=output)
    return model


model = build_model()
model.compile(loss="mean_absolute_percentage_error", optimizer="adam", metrics=['accuracy'])
model.fit([X_train.values]*12, Y_train.values, epochs=2, batch_size=10)
y_pred = model.predict([X_test]*12) 
y = Y_test.tolist() 

RR(y,y_pred)
MAPE(y,y_pred)
    
# ----Ploting----
import matplotlib.pyplot as plt
plt.plot(y,color='red')
plt.plot(y_pred)
# plt.axis([0, 6, 0, 20])
plt.show()


