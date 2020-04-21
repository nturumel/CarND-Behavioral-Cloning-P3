import argparse
import base64
import json
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from time import gmtime, strftime

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
col, row = 200,66

def crop_image(image):
    shape = image.shape
    #Cut off the sky from the original picture
    crop_up = shape[0]/5

    #Cut off the front of the car
    crop_down = shape[0]-25

    image = image[crop_up:crop_down, 0:shape[1]]
    image = cv2.resize(image,(col,row), interpolation=cv2.INTER_AREA)
    return image



sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    images=[]
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
    image_array_cropped= crop_image(image_array)

    transformed_image_array = image_array_cropped[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.12
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    time.replace(':','_').replace('-','_')
    name = 'IMG_realtime/image'+ time +'.jpg'
    cv2.imwrite(name,image)

    return images






@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
