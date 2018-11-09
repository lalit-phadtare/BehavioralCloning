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
[image6]: ./images/loss_plot_general.png "Loss plots during training"
[image7]: ./images/layer1_t1_out.png "Layer 1 viz. for track1 sample"
[image8]: ./images/layer1_t2_out.png "Layer 1 viz. for track1 sample"
[image9]: ./images/layer2_t1_out.png "Layer 2 viz. for track1 sample"
[image10]: ./images/layer2_t2_out.png "Layer 2 viz. for track1 sample"
[image11]: ./images/layer3_t1_out.png "Layer 3 viz. for track1 sample"
[image12]: ./images/layer3_t2_out.png "Layer 3 viz. for track1 sample"
[image13]: ./images/layer4_t1_out.png "Layer 4 viz. for track1 sample"
[image14]: ./images/layer4_t2_out.png "Layer 4 viz. for track1 sample"
[image15]: ./images/layer5_t1_out.png "Layer 5 viz. for track1 sample"
[image16]: ./images/layer5_t2_out.png "Layer 5 viz. for track1 sample"
[image17]: ./images/center.jpg "Sample input from tack1"
[image18]: ./images/center_t2.jpg "Sample input from tack2"


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
* video.mp4 which is a video of autonomous driving on track1
* video1.mp4 which is a video of autonomous driving on track2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 ([model.py lines 81-93](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L81-L93))

The model includes RELU layers to introduce nonlinearity (code line 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77). 

#### 2. Attempts to reduce overfitting in the model

The given driving data was limited so to expose the model to different data, I tried the collecting data as discussed next. The training and validation losses were plotted and found to be in reasonable range before testing on autonomous mode.

The number of epochs was set to 3 as the loss leveled off after 3rd epoch.

The final loss decline in train and valid data is plotted here:

![alt text][image6]


#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually ([model.py line 96](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L96)).

#### 4. Appropriate training data

The final training data I used was collected by me driving the car in the simulator in laps following order:
1. 2 laps in one direction with car in center.
2. one lap, recording car coming from off tracks to the center of the lane.
3. one lap driving close to the marked yellow lines.
4. repeat step one to 3 for both tracks


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started off the model discussed and shown as the more powerful network which is a Nvidia model. This is the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which shows the model.

I started off with the given driving data. I synthesized more data using approaches like flipping images and add measurement corrected left and right images. I will discuss this in detail further.

I used the generator approach to reduce memory usage by reading data in batches into the memory as needed. Further the I split the data in training and validation to keep a tap on train and validation losses. 

The optimizer I chose was "Adam's" and measure mean squared error, "MSE" losses.

The first run showed satisfactory MSE values however during the autonomous run, the car had a difficult time handling the turns and went of the driving range and did not recover.

Since the model is well established, I focused my efforts on collecting more and variable data. I finally collected data as shown further.


#### 2. Final Model Architecture

The final model architecture ([model.py lines 81-93](https://github.com/lx-px/BehavioralCloning/blob/master/model.py#L81-L93)) consisted of a convolution neural network with the following layers and layer sizes:

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


##### Model conv. layer visualization for both tracks:

Sample input used for visualization from track1:

![alt text][image17]

Sample input used for visualization from track2:

![alt text][image18]

Layer 1 track 1 output:

![alt text][image7]

Layer 1 track 2 output:

![alt text][image8]

Layer 2 track 1 output:

![alt text][image9]

Layer 2 track 2 output:

![alt text][image10]


Layer 3 track 1 output:

![alt text][image11]

Layer 3 track 2 output:

![alt text][image12]

Layer 4 track 1 output:

![alt text][image13]

Layer 4 track 2 output:

![alt text][image14]


Layer 5 track 1 output:

![alt text][image15]

Layer 5 track 2 output:

![alt text][image16]


It seems to me that layer 1-3 are good at identifying the lane markings in both tracks and the later layers are picking up some finer detail (which are not very clear to me)


#### 3. Creation of the Training Set & Training Process

To expose the model to different driving situations I manually collected data by driving the car in laps for both tracks.

First for both tracks, I made two laps at a steady speed trying hard to center the car in the lane. 
Next, to recover from situations where the car crosses the lanes while on curves,  I moved the car on to the curves and over the lane markings multiple times and recorded the instants where I then get the car back to center. I did this for both inner and outer curves.
Providing learning data for moments where the car tends to go off track is very important in my experience.

I also recorded one more lap while driving the car close to the lanes markings.

To provide equal left and right turns, I flipped the image and steering measurements for every center camera image and appended it to the the data set.

To provide a wide range of steering angles and views, I used the left and right camera views. Their measurements were adjusted by adding and subtracting a correction factor of 0.2.

The data set for each track was split into training and validation set based on 80/20% rule respectively. 

The train data from both tracks was combined and shuffled to avoid temporal bias and then fed to the model in batches of 32 samples. 

Number of epochs were set to 3.

The MSE losses were:
For train: 0.0737
For validation: 0.0693


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

