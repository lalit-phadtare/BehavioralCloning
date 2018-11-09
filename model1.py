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

dataDir = 'mydatatrack1/'
dataDir1 = 'mydatatrack2_1/'


def generator(df, bsize=32):
    "generator for training and validation samples"
    
    nsize = len(df)
    while 1:
        
        for offset in range(0, nsize, bsize):
            images = []
            measurements = []
            for ii in range(offset, min(offset+bsize-1, nsize)):
                steering_center = float(df.iloc[ii,3])
            
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                #img_center = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 0].split('\\')[-1]), cv2.COLOR_BGR2RGB)
                #img_left = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 1].split('\\')[-1]), cv2.COLOR_BGR2RGB)
                #img_right = cv2.cvtColor(cv2.imread(dataDir + 'IMG/' + df.iloc[ii, 2].split('\\')[-1]), cv2.COLOR_BGR2RGB)
                
                img_center = cv2.cvtColor(cv2.imread(df.iloc[ii, 0]), cv2.COLOR_BGR2RGB)
                img_left = cv2.cvtColor(cv2.imread(df.iloc[ii, 1]), cv2.COLOR_BGR2RGB)
                img_right = cv2.cvtColor(cv2.imread(df.iloc[ii, 2]), cv2.COLOR_BGR2RGB)
                
                
                #flip centre image and corresponding measurement
                img_center_fl = cv2.flip(img_center,1)
                steering_center_fl = steering_center * -1.0
                
                            
                # add images and angles to data set
                images.extend([img_center, img_left, img_right, img_center_fl])
                measurements.extend([steering_center, steering_left, steering_right, steering_center_fl])
            
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield((X_train, y_train))

         


df = pd.read_csv(dataDir + 'driving_log.csv', names=['center', 'left', 'right', 's', 't', 'b', 'd'])
train_df, validation_df = train_test_split(df, test_size=0.2)

df1 = pd.read_csv(dataDir1 + 'driving_log.csv', names=['center', 'left', 'right', 's', 't', 'b', 'd'])
train_df1, validation_df1 = train_test_split(df1, test_size=0.2)

train_df = train_df.append(train_df1)[train_df1.columns.tolist()]
validation_df = validation_df.append(validation_df1)[validation_df1.columns.tolist()]

train_df = shuffle(train_df)


train_generator = generator(train_df, bsize=32)
validation_generator = generator(validation_df, bsize=32)


    
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

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch= \
                 len(train_df)/32, validation_data=validation_generator, \
                 validation_steps=len(validation_df)/32, nb_epoch=3, verbose=1)

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
#plt.plot(hist['loss'])
#plt.plot(hist['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('modelgeneral.h5')


"""
model.load_weights('modeltrack2.h5')


from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

# Testing
test = cv2.cvtColor(cv2.imread('images/center.jpg'), cv2.COLOR_BGR2RGB)
layer_outs = functor([test, 1.])
print(layer_outs)
"""