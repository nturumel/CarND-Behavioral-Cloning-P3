
import numpy as np
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import h5py
from keras import __version__ as keras_version
from train import buildModel
from keras import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

IMG_SIZE = (160, 160)
image = Image.open(R"C:\Users\nihar\Documents\GitHub\CarND-Behavioral-Cloning-P3\data\IMG\center_2020_04_19_21_24_48_878.jpg")
image = np.asarray(image)
image = tf.image.resize(image, IMG_SIZE)
model = buildModel()
model.load_weights(R"C:\Users\nihar\Documents\GitHub\CarND-Behavioral-Cloning-P3\steering_prediction_model.h5")
steering_angle = float(model.predict(image[None, :, :, :], batch_size=1))
print(steering_angle)
