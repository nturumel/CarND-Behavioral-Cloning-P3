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

[image1]: "./examples/center_2020_04_19_21_24_48_878.jpg" "Image Visualization"
[image2]: ./examples/center_2020_04_19_21_24_48_878_cropped.jpg "Cropping"
[image3]: ./examples/center_2020_04_19_21_24_48_878_cropped_flipped.jpg "Flipped"
[image4]: ./examples/left_2020_04_19_21_24_48_878.jpg "Left"
[image5]: ./examples/right_2020_04_19_21_24_48_878.jpg "Right Image"
[image6]: ./examples/right_2020_04_19_21_24_48_878.jpg "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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
##### 1. A normalisation layer.
It consists of:model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

##### 2. A cropping layer.
model.add(Cropping2D(cropping=((50,20),(0,0))))

##### 3. A CNN layer of size of 24 x 5 x 5 followed by relu activation.
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

##### 3. A Maxpooling layer of size 2 x 2.
model.add(MaxPooling2D(pool_size=(2, 2)))

##### 4. A Dropout Layer.
model.add(Dropout(0.5))

##### 5. A CNN layer of size of 36 x 5 x 5 followed by relu activation.
model.add((Convolution2D(36,5,5,subsample=(2,2),activation="relu")))

##### 6. A Dropout Layer.
model.add(Dropout(0.5))

##### 7. A CNN layer of size of 36 x 5 x 5 followed by relu activation.
model.add(Convolution2D(63,3,3,activation="relu"))

##### 8. A Dropout Layer.
model.add(Dropout(0.5))

##### 9. A CNN layer of size of 36 x 5 x 5 followed by relu activation.
model.add(Convolution2D(63,3,3,activation="relu"))

##### 10. A Dropout Layer.
model.add(Dropout(0.5))

##### 11. A Flattening Layer.
model.add(Flatten())

##### 12. A Series of Fully Connected Layer.
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

##### 13. Output (Predicted angle).
model.add(Dense(1))


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

I recorded laps in the forward and backward direction to prevent a specific side from dominating the training data.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
