import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

import json
from keras.models import model_from_json
import sys
import os

os.chdir(sys.path[0])



json_file = open(r"C:\Users\Nihar\Documents\CarND-Behavioral-Cloning-P3\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"C:\Users\Nihar\Documents\CarND-Behavioral-Cloning-P3\model.h5")
print("Loaded model from disk")

# serialize model to JSON
model_json = loaded_model.to_json()
with open("model_woweights.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save("model_woweights.h5")
print("Saved model to disk")
