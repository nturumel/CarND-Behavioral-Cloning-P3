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

[image1]: ./examples/center_2020_04_19_21_24_48_878.jpg "Image Visualization"
[image2]: ./examples/center_2020_04_19_21_24_48_878_cropped.jpg "Cropping"
[image3]: ./examples/center_2020_04_19_21_24_48_878_cropped_flipped.jpg "Flipped"
[image4]: ./examples/left_2020_04_19_21_24_48_878.jpg "Left"
[image5]: ./examples/right_2020_04_19_21_24_48_878.jpg "Right Image"

#### 1. Project includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Project includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Project code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on CNN architechture developed by the autonmous driving team at NVIDIA.
It consists of:
| Layer                                                             | (type)       | Description |        |      |     |
|-------------------------------------------------------------------|--------------|-------------|--------|------|-----|
| ================================================================= |              |             |        |      |     |
| lambda_1                                                          | (Lambda)     | (None,      | 160,   | 320, | 3)  |
| _________________________________________________________________ |              |             |        |      |     |
| cropping2d_1                                                      | (Cropping2D) | (None,      | 90,    | 320, | 3)  |
| _________________________________________________________________ |              |             |        |      |     |
| conv2d_1                                                          | (Conv2D)     | (None,      | 24,    | 5,   | 5)  |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_1                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| conv2d_2                                                          | (Conv2D)     | (None,      | 36,    | 5,   | 5)  |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_2                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| conv2d_3                                                          | (Conv2D)     | (None,      | 36,    | 5,   | 5)  |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_3                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| conv2d_4                                                          | (Conv2D)     | (None,      | 36,    | 5,   | 5)  |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_4                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| conv2d_5                                                          | (Conv2D)     | (None,      | 36,    | 5,   | 5)  |
| _________________________________________________________________ |              |             |        |      |     |
| max_pooling2d_1                                                   | (MaxPooling2 | (None,      | 21,    | 79,  | 24) |
| _________________________________________________________________ |              |             |        |      |     |
| dropout_1                                                         | (Dropout)    | (None,      | 21,    | 79,  | 24) |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_5                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| flatten_1                                                         | (Flatten)    | (None,      | 10710) |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_6                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| dense_1                                                           | (Dense)      | (None,      | 100)   |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_7                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| dense_2                                                           | (Dense)      | (None,      | 50)    |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_8                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_9                                                      | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| dense_3                                                           | (Dense)      | (None,      | 10)    |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| Activation_10                                                     | (Activation) | (Relu)      |        |      |     |
| _________________________________________________________________ |              |             |        |      |     |
| dense_4                                                           | (Dense)      | (None,      | 1)     |      |     |          11


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

I recorded laps in the forward and backward direction to prevent a specific side from dominating the training data.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, 
I manually tuned the optimiser (train.py line 69) and tried different values between 0.0001 and 0.001. I settled on 0.001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ensure that the model completed a successful lap around the simulator.

My first step was to use a convolution neural network model similar to the one developed by the autonomous driving team at NVIDIA.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20).

To combat the overfitting, I modified the model by adding dropout layers.

Then I recorded multiple laps in both directions to prevent bias. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, I recorded a few more laps focused specifically on recovering vehicle once it has veered towards the side.
I also improved my augmenting process by flipping each image and inverting the angle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

Here is an example image of center lane driving:

![alt text][image1]

I also made use of the left and right side camera images by adding and subtracting an angle of 0.2 from each of them respectively .

![alt text][image4]
![alt text][image5]

I also noticed that the pertainaint data required for predicting angles lay in the middle of the image, so I added a cropping layer.
![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would provide me with instances of opposite velocity. For example, here is an image that has then been flipped.

I was facing problem at a specific spot after the bridge the car went straight ahead into the dirt road, I realised that this was happening beacuse a baricade was absent and that it was not differentiating between the road and the dirt, so in order to work around that I added a random brightness augmentation.

Initially after adding the augmentation I was unable to get good convergence for my model, as it had too many pooling and dropout layers, I switched to ELU activation but that was not helpful either.

I finally dropped a number of pooling and dropout layers and attached a single dropout layer at the end of my convulational framework after a maxpooling layer.

This lead to good convergeance.

![alt text][image3]

After the collection process, I had more than 33000 number of data points. I then preprocessed this data by normalising it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

#### Adaptative throttle.
I noticed that sometimes my simulator was sluggish and it had a hard time working out sharp turns. So I decided to manupulate the throttle based on the steering angle. Larger the steering angle lower the speed.

#### Video
[video compilation](https://youtu.be/uSToEkrpm14)


