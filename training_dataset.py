import sys
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy 
from keras.optimizers import Adam  
from keras.regularizers import l2  


data=pd.read_csv("fer2013.csv")


x_train,y_train,x_test,y_test=[],[],[],[]

for ind,cat in data.iterrows():
    image=cat["pixels"].split(" ")
    if cat["Usage"]=="Training":
        x_train.append(np.array(image,"float32"))
        y_train.append(cat["emotion"])

    elif cat["Usage"]=="PublicTest":
        x_test.append(np.array(image,"float32"))
        y_test.append(cat["emotion"])


# for ind,cat in data_test.iterrows():
#     image=cat["pixels"].split(" ")
#     x_test.append(np.array(image,"float32"))
#     y_test.append(cat["emotions"])

num_features=64
num_labels=7
batch_size = 64  
epochs = 30  
width =  48
height =  48 

x_train = np.array(x_train,'float32')  
y_train = np.array(y_train,'float32')  
x_test = np.array(x_test,'float32')  
y_test = np.array(y_test,'float32')  
  
y_train=np_utils.to_categorical(y_train, num_classes=num_labels)  
y_test=np_utils.to_categorical(y_test, num_classes=num_labels)  

x_train -= np.mean(x_train, axis=0)  
x_train /= np.std(x_train, axis=0)  
  
x_test -= np.mean(x_test, axis=0)  
x_test /= np.std(x_test, axis=0)  
  
x_train = x_train.reshape(x_train.shape[0], height, width, 1)  
  
x_test = x_test.reshape(x_test.shape[0], height, width, 1)



model =Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1:])))  
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
model.add(Dropout(0.5))  


model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(Conv2D(64, (3, 3), activation='relu'))   
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
model.add(Dropout(0.5))  
  

model.add(Conv2D(128, (3, 3), activation='relu'))  
model.add(Conv2D(128, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
  
model.add(Flatten()) 


model.add(Dense(1024, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(1024, activation='relu'))  
model.add(Dropout(0.2))  
  
model.add(Dense(num_labels, activation='softmax'))  



model.compile(loss=categorical_crossentropy,  
              optimizer=Adam(),  
              metrics=['accuracy'])  


model.fit(x_train, y_train,  
          batch_size=batch_size,  
          epochs=epochs,  
          verbose=1,  
          validation_data=(x_test, y_test),  
          shuffle=True)  



op_json = model.to_json()  
with open("trained.json", "w") as json_file:  
    json_file.write(op_json)  
model.save_weights("trained.h5") 
