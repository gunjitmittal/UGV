from builtins import print
from distutils.command.upload import upload
import os
import pandas as pd
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import trainingHelper 
from PIL import Image
# import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import date, datetime


#step 1 - Initialize data
path = 'DataCollected'   
initial_time = datetime.now()

#Import Data Info
data = trainingHelper.importDataInfo(path)
print("h&&&&&&&&&&&&&&&&&&&&Data saved as dataframe&&&&&&&&&&&&&&&&")
# print(data.head())

#step 2 - Visualize and balance data
data = trainingHelper.balanceData(data,display = True)
# print(data.head())
print("<<<<<<<<<<<<<<<<<<<Visualization and Balancing Done>>>>>>>>>>>>>>>>>>>>>>>")

#step 3 - prepare for processing
imagesPath, steerings = trainingHelper.loadData(path, data)
#print('No of Path created for images', len(imagesPath), len(steerings))
#cv2.imshow('Test image', cv2.imread(imagesPath[5]))
#cv2.waitkey(0)
print("<<<<<<<<<<<<<<<<<<<<<<<<<<<Array of imagepath and steering done>>>>>>>>>>>>>>>>>>>>>>>>.\n")
print("")
#step 4 - Split for training and validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings,
                                             test_size = 0.2, random_state = 10) #random_state set to an integer value for getting same set of data each time

print('Total training Images:',len(xTrain))
print('Total validation Images:',len(xVal))
# xTrain = xTrain.reshape(-1, 240,120, 1)
# print(xTrain.size)



#step 5 - Augment data
# done in step 8 only for training data
#step 6 - pre-process


#step 7 - create model 
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#TO TRY NVIDIA MODEL UNCOMMENT BELOW LINE
#model = trainingHelper.nvidia_model()

#TO TRY TRANSFER LEARNING MODEL
model = trainingHelper.transfer_learning()

#TO TRY ORIGINAL MODEL
#model = trainingHelper.create_model()
# we also have to change the image size from (224,224,3) to (66,200,3) to train on nvidia or original model

print("###############################")
# use pretrained Model
#model.load_weights('/Users/abhishekkumar/Desktop/abhishek/pretrained_model.h5')

filepath = 'Transfer_model64_10.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=True, save_weights_only=False, \
                             mode='auto', save_frequency=1)#ModelCheckpoint callback is used in conjunction with training using model. fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.




#step 8 - Training on cpu with constant epoch number
# I
batch_size = 32
epoch =50
Train_time = datetime.now()
dataloader_train=trainingHelper.dataGen(xTrain,yTrain,batch_size,1)
print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
dataloader_val=trainingHelper.dataGen(xVal,yVal,16,0)
print(">>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
History = model.fit(dataloader_train,
                   steps_per_epoch =len(xTrain)//batch_size,
                   epochs = epoch,
                   validation_data = dataloader_val,
                   validation_steps = 1,
                   callbacks=[checkpoint])

print("########################")

#step 9 - Save the model
#print("************") 

model.save('Transfer_model64_10.h5') 
print('Model Saved')
Train_time = (datetime.now() - Train_time).total_seconds()
print("training time: ",Train_time)
file1=open('../finalanalysis/constepochcpu.dat','w')
file1.write(str(Train_time)+'\n')
temp = trainingHelper.uploadDate(batch_size,epoch,Train_time)
print("done uploading data.")
#step 10 - plot the results

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.savefig("Transfer_model_result3250cpu.png")
#plt.show()                                                                                        

#II
#step 8 - Training
batch_size = 64
epoch =50
Train_time = datetime.now()
dataloader_train=trainingHelper.dataGen(xTrain,yTrain,batch_size,1)
print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
dataloader_val=trainingHelper.dataGen(xVal,yVal,16,0)
print(">>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
History = model.fit(dataloader_train,
                   steps_per_epoch =len(xTrain)//batch_size,
                   epochs = epoch,
                   validation_data = dataloader_val,
                   validation_steps = 1,
                   callbacks=[checkpoint])

print("########################")

#step 9 - Save the model
#print("************") 

model.save('Transfer_model64_10.h5') 
print('Model Saved')
Train_time = (datetime.now() - Train_time).total_seconds()
print("training time: ",Train_time)
file1.write(str(Train_time)+'\n')
temp = trainingHelper.uploadDate(batch_size,epoch,Train_time)
print("done uploading data.")
#step 10 - plot the results

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.savefig("Transfer_model_result6450cpu.png")

#III
#step 8 - Training
batch_size = 128
epoch =50
Train_time = datetime.now()
dataloader_train=trainingHelper.dataGen(xTrain,yTrain,batch_size,1)
print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
dataloader_val=trainingHelper.dataGen(xVal,yVal,16,0)
print(">>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
History = model.fit(dataloader_train,
                   steps_per_epoch =len(xTrain)//batch_size,
                   epochs = epoch,
                   validation_data = dataloader_val,
                   validation_steps = 1,
                   callbacks=[checkpoint])

print("########################")

#step 9 - Save the model
#print("************") 

model.save('Transfer_model64_10.h5') 
print('Model Saved')
Train_time = (datetime.now() - Train_time).total_seconds()
print("training time: ",Train_time)
file1.write(str(Train_time)+'\n')
file1.close()
temp = trainingHelper.uploadDate(batch_size,epoch,Train_time)
print("done uploading data.")
#step 10 - plot the results

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.savefig("Transfer_model_result12850cpu.png")

#Part2 : constant batch size
#step 8 - Training
batch_size = 128
epoch =100
Train_time = datetime.now()
dataloader_train=trainingHelper.dataGen(xTrain,yTrain,batch_size,1)
print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
dataloader_val=trainingHelper.dataGen(xVal,yVal,16,0)
print(">>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
History = model.fit(dataloader_train,
                   steps_per_epoch =len(xTrain)//batch_size,
                   epochs = epoch,
                   validation_data = dataloader_val,
                   validation_steps = 1,
                   callbacks=[checkpoint])

print("########################")

#step 9 - Save the model
#print("************") 

model.save('Transfer_model64_10.h5') 
print('Model Saved')
Train_time = (datetime.now() - Train_time).total_seconds()
print("training time: ",Train_time)
file2=open('../finalanalysis/constantbatchcpu.dat','w')
file2.write(str(Train_time)+'\n')
temp = trainingHelper.uploadDate(batch_size,epoch,Train_time)
print("done uploading data.")
#step 10 - plot the results

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.savefig("Transfer_model_result128100cpu.png")


#step 8 - Training
batch_size = 128
epoch =150
Train_time = datetime.now()
dataloader_train=trainingHelper.dataGen(xTrain,yTrain,batch_size,1)
print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
dataloader_val=trainingHelper.dataGen(xVal,yVal,16,0)
print(">>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
History = model.fit(dataloader_train,
                   steps_per_epoch =len(xTrain)//batch_size,
                   epochs = epoch,
                   validation_data = dataloader_val,
                   validation_steps = 1,
                   callbacks=[checkpoint])

print("########################")

#step 9 - Save the model
#print("************") 

model.save('Transfer_model64_10.h5') 
print('Model Saved')
Train_time = (datetime.now() - Train_time).total_seconds()
print("training time: ",Train_time)
file2.write(str(Train_time)+'\n')
file2.close()
temp = trainingHelper.uploadDate(batch_size,epoch,Train_time)
print("done uploading data.")
#step 10 - plot the results

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.legend(['Training','validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.savefig("Transfer_model_result128150cpu.png")
