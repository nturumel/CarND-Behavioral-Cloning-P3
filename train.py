import os
import csv
import sys
import augment

os.chdir(sys.path[0])

augment.augment()

# load the samples csv
samples=[]
with open(r".\modified.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit as shuffle
# split the samples data set by 80 20 ratio
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

import cv2
import numpy as np
import sklearn
# create a generator that loads the images
def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1:
        # shuffle the num_samples
        shuffle(samples)

        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            angles=[]
            # iterate through each line in batch samples
            # and append to list
            for batch_sample in batch_samples:
                # load the image
                name=r"./data/IMG/"+batch_sample[0].split('/')[-1]
                name=name.strip()
                img=cv2.imread(name)
                angle=batch_sample[1]
                images.append(img)
                angles.append(angle)
                img_flip_lr = cv2.flip(img, 1)
                images.append(img_flip_lr)
                angles.append(str(-float(angle)))

            X_train=np.array(images)
            y_train=np.array(angles)


            yield sklearn.utils.shuffle(X_train,y_train)

train_generator=generator(train_samples)
validation_generator=generator(validation_samples)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Convolution2D, MaxPooling2D,Dropout
from keras.layers.core import Dense,Activation,Flatten,Lambda
from keras.layers import Lambda
from math import ceil
from keras import optimizers

adam = optimizers.Adam(lr=0.001)

batch_size=32
model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add((Convolution2D(36,5,5,subsample=(2,2),activation="relu")))
model.add(Dropout(0.5))
model.add(Convolution2D(63,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(63,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer=adam,metrics=['accuracy'])
model.fit_generator(train_generator,
steps_per_epoch=ceil(len(train_samples)/batch_size),
validation_data=validation_generator,
validation_steps=ceil(len(validation_samples)/batch_size),
epochs=20,verbose=1)

print("Completed Training")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")
