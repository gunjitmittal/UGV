from email import header
import os
#from tkinter import _Padding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
#from tkinter import _Padding
import random

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

#############
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob
###############
#### STEP 1 - INITIALIZE DATA
def getName(filePath):
    print(type(filePath))
    myImagePathL = filePath.split('/')[-2:]
    #print(myImagePathL)
    myImagePath = os.path.join(myImagePathL[0],myImagePathL[1])
    print(myImagePath)
    return myImagePath

def importDataInfo(path):
    columns = ['Center','steering_val']
    noOfFolders = len(os.listdir(path))//2
    data = pd.DataFrame()
    for x in range(0,1):
        dataNew = pd.read_csv(os.path.join(path, f'n_log_full.csv'))
        #print(f'{x}:{dataNew.shape[0]} ',end='')
        #### REMOVE FILE PATH AND GET ONLY FILE NAME
        #print(getName(dataNew['Center'][0]))
        dataNew['Center']=dataNew['Center'].apply(getName)
        data =data.append(dataNew,True)
    print(' ')
    print('Total Images Imported',data.shape[0])
    return data

#### STEP 2 - VISUALIZE AND BALANCE DATA
def balanceData(data,display=True):
    nBin = 31
    samplesPerBin =  300
    hist, bins = np.histogram(data['steering_val'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering_val']), np.max(data['steering_val'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['steering_val'])):
            if data['steering_val'][i] >= bins[j] and data['steering_val'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    if display:
        hist, _ = np.histogram(data['steering_val'], (nBin))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['steering_val']), np.max(data['steering_val'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    return data

#### STEP 3 - PREPARE FOR PROCESSING
def loadData(path, data):
  imagesPath = []
  steering = []
  #print(len(data))
  for i in range(0,len(data)):
    indexed_data = data.iloc[i]
#-------> use below for testbed data
    filename=path+"/"+ indexed_data[0].split('/')[1]
#-------->use below for simulator data
    #print(indexed_data[0].split('IMG\\')[1])
    #filename = "/Users/abhishekkumar/Desktop/abhishek/DataCollected_simulator/IMG/"+ indexed_data[0].split('IMG\\')[1]
    filename = filename.replace("\\","/")
    print(filename)
    imagesPath.append(filename)
#imagesPath.append( os.path.join(path,indexed_data[0]))
    steering.append(float(indexed_data[1]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering


#### STEP 5 - AUGMENT DATA
def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

# imgRe,st = augmentImage('DataCollected/IMG18/Image_1601839810289305.jpg',0)
# #mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

#### STEP 6 - PREPROCESS
def preProcess(img):
    #img = img[54:120,:,:]
    img = img[60:-25, :, :] #change done
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    # resize image for transferlearning:224,244
    # resize image for create model : 66,200
    img = cv2.resize(img, (224, 224))
    img = img/255
    return img

# imgRe = preProcess(mpimg.imread('DataCollected/IMG18/Image_1601839810289305.jpg'))
# # mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

####STEP 7 - CREATE MODEL
############ Create Model using Sequential learning###############
def createModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0 , input_shape =(66, 200, 3)))  # changes done
    model.add(Convolution2D(24, (5, 5), (2, 2), activation='elu')) #input_shape=(66, 200, 3)
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))#dot product of weights and input array
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5))  # changes done
    model.add(Flatten())
    # model.add(Flatten(input_shape = (66,200,3)))
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))
        # lr = 3xe-4
    model.compile(Adam(learning_rate=1.0e-4),loss= 'MSE') 
    
    model.summary()
    return model

#********************Nvidia Model ****************
def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    from keras.backend import tf as ktf
    return ktf.image.resize(img, [66, 200])


def nvidia_model():
    model = Sequential()
    # Cropping image
    model.add(Lambda(lambda imgs: imgs[:,80:,:,:], input_shape=(66, 200, 3)))
    # Normalise the image - center the mean at 0
    model.add(Lambda(lambda imgs: (imgs/255.0) - 0.5))
    model.add(Lambda(resize))

    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Convolution2D(24, (5, 5), (2,2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, (5, 5),(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(48, (5, 5),(2, 2), activation='relu'))    
    model.add(BatchNormalization())

    
    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Convolution2D(64, (3, 3),(1, 1), activation='relu')) 
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3, 3),(1, 1), activation='relu')) 
    model.add(BatchNormalization())
    
    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())
    # model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(1164, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(50, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(loss = "MSE", optimizer = Adam(learning_rate = 0.001))
    return model 
##################################################
################Transfer Learning Model#####################
def transfer_learning():
    InceptionV3_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # The last 15 layers fine tune
    #window size is 3x3
    for layer in InceptionV3_model.layers[:-15]:
        layer.trainable = False

    x = InceptionV3_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units=50, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units=10, activation='relu')(x)
    #x = Dropout(0.3)(x)
    output  = Dense(units=1)(x)
    model = Model(InceptionV3_model.input, output)
    model.compile(Adam(lr=1.0e-4),loss= "MSE" )# tf.keras.metrics.mean_squared_error) 

    model.summary()
    return model
    #########################

#### STEP 8 - TRAINING

def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = np.asarray(img)
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))

def uploadDate(batch_size,epoch,totaltime):
    if os.path.exists('timeanalysis_CPU.csv'):
        df = pd.read_csv('timeanalysis_CPU.csv')
        df = df.append({"batch_size":batch_size,"epoch":epoch,"time(sec)":totaltime},ignore_index=True)
        # df.loc[len(df)-1]=
        # df = df.append([batch_size,epoch,totaltime],ignore_index=True)

        df.to_csv('timeanalysis_CPU.csv',index=False)
        print("done")
        return True
    else:
        # data = [batch_size,epoch,totaltime]
        df = pd.DataFrame(columns=(["batch_size","epoch","time(sec)"]))
        df.to_csv('timeanalysis_CPU.csv',index=False)
        a = uploadDate(batch_size,epoch,totaltime)
        return a
