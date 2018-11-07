# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.jpg "Center image"
[image2]: ./images/recovery.jpg "Recovery Image"
[image3]: ./images/left.jpg "Left Camera Image"
[image4]: ./images/normal.jpg "Normal Image"
[image5]: ./images/flipped.jpg "Flipped Image"
[image6]: ./images/loss_plot.png "Loss plots during training"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 ([model.py lines 76-88](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L76-L88))

The model includes RELU layers to introduce nonlinearity (code line 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The given driving data was limited so to expose the model to different data, I tried the collecting data as discussed next. The training and validation losses were plotted and found to be in reasonable range before testing on autonomous mode.

The number of epochs was set to 3 as the loss leveled off after 3rd epoch.

The final loss decline in train and valid data is plotted here:

![alt text][image6]


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually ([model.py line 91](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L91)).

#### 4. Appropriate training data

The final training data I used was collected by me driving the car in the simulator in laps following order:
1. 2 laps in one direction with car in center.
2. one lap, recording car coming from off tracks to the center of the lane.
3. one lap driving close to the marked yellow lines.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started off the model discussed and shown as the more powerful network which is a Nvidia model. This is the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which shows the model.

I started off with the given driving data. I synthesized more data using approaches like flipping images and add measurement corrected left and right images. I will discuss this in detail further.

I used the generator approach to reduce memory usage by reading data in batches into the memory as needed. Further the I split the data in training and validation to keep a tap on train and validation losses. 

The optimizer I chose was "Adam's" and measure mean squared error, "MSE" losses.

The first run showed satisfactory MSE values however during the autonomous run, the car had a difficult time handling the turns and went of the driving range and did not recover.

Since the model is well established, I focused my efforts on collecting more and variable data. I finally collected data as shown further.


#### 2. Final Model Architecture

The final model architecture ([model.py lines 76-88](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L76-L88)) consisted of a convolution neural network with the following layers and layer sizes:

| Layer no.     | Layer type           | Layer specs.                                              | 
| ------------- | ---------------------| --------------------------------------------------------- | 
| 1             |      Input           |       size = 160x320x3                                    |   
| 2             |      Lambda          | Normalization   in range (-0.5, 0.5)                      |  
| 3             |      2D-Cropping     | crop 50 top and 20 bottom rows                            |   
| 4             |      2D-Convolution  | kernel = 24x5x5, strides = (2,2), activation = RELU       | 
| 5             |      2D-Convolution  | kernel = 36x5x5, strides = (2,2), activation = RELU       | 
| 6             |      2D-Convolution  | kernel = 48x5x5, strides = (2,2), activation = RELU       |
| 7             |      2D-Convolution  | kernel = 64x3x3, activation = RELU                        |  
| 8             |      2D-Convolution  | kernel = 64x3x3, activation = RELU                        |   
| 9             |      Flatten         |       100 neurons                                         |   
| 10            |      Dense           |       50 neurons                                          |    
| 11            |      Dense           |       10 neurons                                          | 
| 12            |      Dense           |       1 neurons                                           |   




#### 3. Creation of the Training Set & Training Process

To expose the model to different driving situations I manually collected data by driving the car in laps.

First I made two laps at a steady speed trying hard to center the car in the lane. 
Next, to recover from situations where the car crosses the lanes while on curves,  I moved the car on to the curves and over the lane markings multiple times and recorded the instants where I then get the car back to center. I did this for both inner and outer curves.

I also recorded one more lap while driving the car close to the lanes markings.

To provide equal left and right turns, I flipped the image and steering measurements for every center camera image and appended it to the the data set.

To provide a wide range of steering angles and views, I used the left and right camera views. Their measurements were adjusted by adding and subtracting a correction factor of 0.2.

The data set was shuffled to avoid temporal bias. 

Data was further split into 80% training and 20% validation and fed into the model in batches of 32 samples. 

Number of epochs were set to 3.

The MSE losses were:
For train: 0.0144
For validation: 0.0136


Here are some examples of training images:

Center camera image:

![alt text][image1]


Left camera image:

![alt text][image3]


Car recovering from outside of lanes:

![alt text][image2]


Normal center camera image and it's flipped image:

![alt text][image4]

![alt text][image5]

