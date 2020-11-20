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
[image6]: ./examples/result.gif "result"


#### 1. Project includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* steering_prediction_model.h5 containing a trained convolution neural network 
* ReadMe.md

#### 2. Project includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model
```

#### 3. Project code is usable and readable

The preprocess.py generates filenames_angles.csv used for training, train.py trains on images in **'.data\IMG\'** and produces **steering_prediction_model.h5**.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

| Layer (type)                                                                                       	| Output Shape           	| Param #                    	| Connected to                   	|   	|
|----------------------------------------------------------------------------------------------------	|------------------------	|----------------------------	|--------------------------------	|---	|
| ================================================================================================== 	|                        	|                            	|                                	|   	|
| input_2 (InputLayer)                                                                               	| [(None, 160, 160, 3) 0 	|                            	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| sequential (Sequential)                                                                            	| (None, 160, 160, 3)    	| 0                          	| input_2[0][0]                  	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| mobilenetv2_1.00_160 (Functiona (None, 5, 5, 1280)                                                 	| 2257984                	| sequential[0][0]           	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| global_average_pooling2d (Globa (None, 1280)                                                       	| 0                      	| mobilenetv2_1.00_160[0][0] 	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| dropout (Dropout)                                                                                  	| multiple               	| 0                          	| global_average_pooling2d[0][0] 	|   	|
| dense[0][0]                                                                                        	|                        	|                            	|                                	|   	|
| dense_1[0][0]                                                                                      	|                        	|                            	|                                	|   	|
| dense_2[0][0]                                                                                      	|                        	|                            	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| flatten (Flatten)                                                                                  	| (None, 1280)           	| 0                          	| dropout[0][0]                  	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| dense (Dense)                                                                                      	| (None, 512)            	| 655872                     	| flatten[0][0]                  	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| activation (Activation)                                                                            	| multiple               	| 0                          	| dropout[1][0]                  	|   	|
| dropout[2][0]                                                                                      	|                        	|                            	|                                	|   	|
| dropout[3][0]                                                                                      	|                        	|                            	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| dense_1 (Dense)                                                                                    	| (None, 256)            	| 131328                     	| activation[0][0]               	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| dense_2 (Dense)                                                                                    	| (None, 64)             	| 16448                      	| activation[1][0]               	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| dense_3 (Dense)                                                                                    	| (None, 1)              	| 65                         	| activation[2][0]               	|   	|
| ================================================================================================== 	|                        	|                            	|                                	|   	|
| Total params: 3,061,697                                                                            	|                        	|                            	|                                	|   	|
| Trainable params: 3,027,585                                                                        	|                        	|                            	|                                	|   	|
| Non-trainable params: 34,112                                                                       	|                        	|                            	|                                	|   	|
| __________________________________________________________________________________________________ 	|                        	|                            	|                                	|   	|
| None                                                                                               	|                        	|                            	|                                	|   	|
__________________________________________________________________________________________________


Transfer Learning was used.

    # define key layers
    data_augmentation = tf.keras.Sequential([ tf.keras.layers.experimental.preprocessing.RandomContrast([0, 0.5])])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
    converge = tf.keras.layers.GlobalAveragePooling2D()
    dropout = Dropout(0.2)
    activate = Activation('relu')

    # chaining the layers
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training = False)
    x = converge(x)
    x = dropout(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = dropout(x)
    x = activate(x)
    x = Dense(256)(x)
    x = dropout(x)
    x = activate(x)
    x = Dense(64)(x)
    x = dropout(x)
    x = activate(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, loss='mse')


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I recorded laps in the forward and backward direction to prevent a specific side from dominating the training data.

    datagen = ImageDataGenerator(brightness_range=[0.2, 1.8], channel_shift_range=150.0, validation_split=0.25)
    

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
I later abandoned this as it was not required.
#### Video
![alt text][image6]


