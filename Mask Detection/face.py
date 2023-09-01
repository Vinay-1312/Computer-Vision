# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:19:01 2020

@author: dell
"""


import pandas as pd
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
print(tf.__version__)

#Call the callback function to avoid overfitting
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>0.97):
            print("finished")
            self.model.stop_training=True

callbacks=mycallback()
#preprocecing of images
train_datagen=ImageDataGenerator(rescale=1.0/255,shear_range=0.4,zoom_range=0.2,horizontal_flip=True,rotation_range=40)
train=train_datagen.flow_from_directory("data",target_size=(150,150),class_mode="binary",batch_size=10)
test_datagen=ImageDataGenerator(rescale=1.0/255)
test=test_datagen.flow_from_directory("test",target_size=(150,150),class_mode="binary",batch_size=10)

#CNN Model architecture
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=100,kernel_size=(3,3),activation="relu",input_shape=(150,150,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=100,kernel_size=(3,3),activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))
    
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dropout(rate=0.3))
cnn.add(tf.keras.layers.Dense(50,activation="relu"))
cnn.add(tf.keras.layers.Dense(2,activation="softmax"))
print(cnn.summary())
cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cnn.fit_generator(train,epochs=20,validation_data=test)
video=cv2.VideoCapture(0)


face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    check,img=video.read()
    img=cv2.flip(img,1,1) #Flip to act as a mirror

    
    mini=cv2.resize(img,(img.shape[1]//4,img.shape[0]//4))
    faces=face.detectMultiScale(mini)

    for f in faces:
        (x,y,w,h)=[v*4 for v in f]
        face_img=img[y:y+h,x:x+w]
        resized=cv2.resize(face_img, (150,150))
    
        normalized=resized/255
    
        im=np.reshape(normalized,(1,150,150,3))
        reshpe=np.vstack([im])
        res=cnn.predict(reshpe)
        result=np.argmax(res)
        print(result)
        if result==0:
            label="no mask"
        else:
            label="mask"
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.rectangle(img,(x,y-40),(x+w,y),(0,0,255),-1)
        cv2.putText(img, label, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX ,0.8, (255,255,255),2)
    cv2.imshow("live",img)
    key=cv2.waitKey(0)
    if key==ord("q"):
        break
video.release()
cv2.destroyAllWindows()
