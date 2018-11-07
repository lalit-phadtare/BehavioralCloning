# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 19:04:27 2018

@author: Lalit
"""

import cv2
import csv
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#dir. where the training data is stored
dataDir = 'mydata/'


def generator(df, bsize=32):
    "generator for training/validation samples"
    
    nsize = len(df)
    while 1:
        
        for offset in range(0, nsize, bsize):
            images = []
            measurements = []
            for ii in range(offset, min(offset+bsize-1, nsize)):
                
                #read the cam steering angle
                steering_center = float(df.iloc[ii,3])
            
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                #read the center, left and right images
                img_center = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 0].split('/')[-1]), cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 1].split('/')[-1]), cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 2].split('/')[-1]), cv2.COLOR_BGR2RGB)
                
                #flip centre image and corresponding measurement
                img_center_fl = cv2.flip(img_center,1)
                steering_center_fl = steering_center * -1.0
                
                            
                # add images and angles to data set
                images.extend([img_center, img_left, img_right, img_center_fl])
                measurements.extend([steering_center, steering_left, steering_right, steering_center_fl])
            
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield((X_train, y_train)) #yield the current batch

         

#read the csv file and shuffle the data
df = pd.read_csv(dataDir + 'driving_log.csv')
df = shuffle(df)

#split all the data 80-20 % in test and validation
train_df, validation_df = train_test_split(df, test_size=0.2)

#these generators will read the files, pre-process, create new data in batches using the csv data
bsize = 32
train_generator = generator(train_df, bsize)
validation_generator = generator(validation_df, bsize)

    
#create model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compile and run the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= \
                 len(train_df)/bsize, validation_data=validation_generator, \
                 nb_val_samples=len(validation_df)/bsize, nb_epoch=3, verbose=1)


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save model
model.save('model.h5')
