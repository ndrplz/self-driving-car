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
import os
import numpy as np
from config import *
from model import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image) # frames incoming from the simulator are in RGB format

    # perform preprocessing (crop, resize etc.)
    image_array = preprocess(frame_bgr=cv2.cvtColor(image_array, code=cv2.COLOR_RGB2BGR))
    # plt.imshow(image_array), plt.show()

    # standardization
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


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

    from keras.models import model_from_json

    # load model from json
    json_path ='logs/model.json'
    with open(json_path) as jfile:
        model = model_from_json(jfile.read())

    # load model weights
    weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    # compile the model
    model.compile("adam", "mse")
    #
    # parser = argparse.ArgumentParser(description='Remote Driving')
    # parser.add_argument('model', type=str,
    #                     help='Path to model definition json. Model weights should be on the same path.')
    # args = parser.parse_args()
    # with open(args.model, 'r') as jfile:
    #     # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
    #     # then you will have to call:
    #     #   model = model_from_json(json.loads(jfile.read()))\
    #     # instead.
    #     model = model_from_json(jfile.read())
    #
    # model.compile("adam", "mse")
    # weights_file = args.model.replace('json', 'h5')
    # model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
