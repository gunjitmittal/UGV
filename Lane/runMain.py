import imghdr
from turtle import color
import cv2
import os
from datetime import datetime
import csv
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
from pypylon import pylon
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Model_path = 'Transfer_model.h5'
print("path",Model_path)
model = load_model(Model_path)

print(type(model)) 


#preprocess test data
def preProcess(img):
    if img.any():
        img = img[54:120,:,:]  # changes 54:120 --> 60: -25
        #print(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (3, 3), 0)
        img = cv2.resize(img, (224,224)) #changes (66,200)--> (224,224)
        img = img/255 
        #img = cv2.imshow('IMG',img)
    return img  

def getImg(img,display= True,size=[480,480]):  #changes [480,240] --> [224,224]
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img,(size[0],size[1]))
    if display:
        cv2.imshow('IMG',img)
        print("image display")
    return img

if __name__ == '__main__':
    #display angle for defaule image
    # image_path = '/DataCollected/IMG19/Image_1652948718046751.jpg'
    # imgPath = (('/').join(Model_path.split('/')[:-2]))+image_path
    cap = cv2.VideoCapture('roots.mp4')
    # data = ['IMG','predicted_steering_val']
    mx , mn = 180,-180
    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0,0,0)
        thickness = 1

        #df = pd.DataFrame()
        #arr = []
        #for i in range(5):
        #arr = []
        img = cap.read()
        # print(img)
        # img = cv2.imread(img)
        img = getImg(True)
        frame = img
        # print(img)
        img = np.asarray(img)
        img = preProcess(img)
        #img = cv2.imshow('IMG',img)
        img = np.array([img])
        # print(img)
        norm_steering = (model.predict(img)).astype(np.float)
        # for i in range(5):
        #     arr.append(norm_steering)
        actual_steering = (mx-mn)*((norm_steering+1)/2)+ mn
        print("Actualvalue",actual_steering)
        #arr.append(actual_steering)
        #steering_value = sum(arr)//5
        print(norm_steering)
        Steering_value = str(norm_steering)+"  "+str(actual_steering)
        image = cv2.putText(frame,Steering_value,(0,50), font, fontScale, color, thickness,cv2.LINE_AA)
        cv2.imshow('steering_value Image', image)
        time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    while False:
        # data = importDataInfo(path)
        # imgPath = loadTestData(data)
        imagesPath = '/home/aparna/Desktop/UGV/abhishek/abhishek/'
        #-0.35
        img =  cv2.imread(imagesPath) 
        img = np.asarray(img)
        img = preProcess(img)
        img = np.array([img])
        steering = float(model.predict(img))
        #actual_steering = (mx-mn)*((steering+1)/2) + mn 
        print(steering)
        break
