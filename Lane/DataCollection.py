import cv2 as cv
import pandas as pd
from datetime import datetime
import numpy as np
import os

imgCap = cv.VideoCapture(0)
Steering_value = 0
current_frame = 0
current_time = datetime.now()
df = pd.DataFrame()
arr = []
arr1 =[]
try:
     if not os.path.exists('tihandata'):
        os.makedirs('tihandata')

except OSError:
    print('Error: Creaing directory of data')

imgFrame = 0

while (True):
    ret,frame = imgCap.read()
    if ret:
        seconds = datetime.now()-current_time
        seconds = seconds.total_seconds()
        # frame1 = cv.resize(frame,(1050,1610),interpolation=cv.INTER_CUBIC)
        
        if (seconds>1):
            print("image saved")
            pathImg = '/home/aparna/Desktop/UGV/abhishek/abhishek/Lane/tihandata/IMG'+str(current_frame)+'.jpg'
            cv.imwrite(pathImg,frame)
            arr.append(current_time)
            arr1.append(pathImg)
            current_time = datetime.now()
            current_frame+=1
            dict={'Image':arr1,'Steering_value':arr}
            df = pd.DataFrame(dict)
            df.to_csv('test.csv',index=False)
    if (cv.waitKey(0)==27 & 0xff):
        break
    # cv.imshow('frame1',frame1)
    


imgCap.release()
cv.destroyAllWindows()