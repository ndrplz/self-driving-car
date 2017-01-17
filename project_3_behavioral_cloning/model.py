from os.path import join
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
import keras.backend as K
from config import *


# todo plot distribution of steering angle - solution mirroring?
# todo verify image mirroring with [:, :, ::-1, :]


def split_train_val(csv_driving_data):

    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=0.2, random_state=1)

    return np.array(train_data), np.array(val_data)


def preprocess(frame_bgr, verbose=False):

    # set training images resized shape
    h, w = CONFIG['input_height'], CONFIG['input_width']

    # crop image (remove useless information)
    frame_cropped = frame_bgr[CONFIG['crop_height'], :, :]

    # resize image
    frame_resized = cv2.resize(frame_cropped, dsize=(w, h))

    # eventually convert to grayscale
    if CONFIG['input_channels'] == 1:
        frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(frame_bgr, code=cv2.COLOR_BGR2RGB))
        plt.figure(2), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')


def load_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', normalize=True, augment_data=True):

    # set training images resized shape
    h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

    # prepare output structures
    X_all = np.zeros(shape=(3, batchsize, h, w, c), dtype=np.float32)
    y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
    y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

    # select a random batch
    indexes = np.random.randint(low=0, high=data.shape[0], size=batchsize)

    for i, (ct_path, lt_path, rt_path, steer, throttle, brake, speed) in enumerate(data[indexes]):

        # load and preprocess images
        X_all[0, i] = preprocess(cv2.imread(join(data_dir, ct_path.strip())))
        X_all[1, i] = preprocess(cv2.imread(join(data_dir, lt_path.strip())))
        X_all[2, i] = preprocess(cv2.imread(join(data_dir, rt_path.strip())))

        y_steer[i] = steer
        y_throttle[i] = throttle

    # for each element in the batch, randomly pick one of the three cameras
    # in case the chosen camera is not the central, adjust y_steer accordingly
    delta_correction = CONFIG['delta_correction']
    X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
    for i in range(batchsize):
        camera = random.choice([0, 1, 2])
        X[i] = X_all[camera, i]
        if camera == 1:    # left camera
            y_steer[i] += delta_correction
        elif camera == 2:  # right camera
            y_steer[i] -= delta_correction
        else:              # central camera
            pass

    if augment_data:

        # mirror images with chance=0.5
        if random.choice([True, False]):
            X = X[:, :, ::-1, :]
            y_steer *= -1

        # perturb slightly steering direction
        for i in range(y_steer.shape[0]):
            y_steer[i] += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

        # if color images, randomly change brightness
        for i in range(X.shape[0]): # todo check that this changes X
            # cv2.imshow('pre', np.uint8(cv2.resize(frame, (300, 150))))
            X[i] = cv2.cvtColor(X[i], code=cv2.COLOR_BGR2HSV)
            X[i][:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
            X[i][:, :, 2] = np.clip(X[i][:, :, 2], a_min=0, a_max=255)
            X[i] = cv2.cvtColor(X[i], code=cv2.COLOR_HSV2BGR)
            # cv2.imshow('post', np.uint8(cv2.resize(frame, (300, 150))))
            # cv2.waitKey()

    if normalize:
        # standardize features
        X = (X - np.mean(X)) / np.std(X)

    if K.backend() == 'theano':
        X = X.transpose(0, 3, 1, 2)

    return X, y_steer


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', normalize=True, augment_data=True):

    # set training images resized shape
    h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

    while True:
        # prepare output structures
        X_all = np.zeros(shape=(3, batchsize, h, w, c), dtype=np.float32)
        y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
        y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

        # select a random batch
        indexes = np.random.randint(low=0, high=data.shape[0], size=batchsize)

        for i, (ct_path, lt_path, rt_path, steer, throttle, brake, speed) in enumerate(data[indexes]):

            # load and preprocess images
            X_all[0, i] = preprocess(cv2.imread(join(data_dir, ct_path.strip())))
            X_all[1, i] = preprocess(cv2.imread(join(data_dir, lt_path.strip())))
            X_all[2, i] = preprocess(cv2.imread(join(data_dir, rt_path.strip())))

            y_steer[i] = steer
            y_throttle[i] = throttle

        # for each element in the batch, randomly pick one of the three cameras
        # in case the chosen camera is not the central, adjust y_steer accordingly
        delta_correction = CONFIG['delta_correction']
        X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
        for i in range(batchsize):
            camera = random.choice([0, 1, 2])
            X[i] = X_all[camera, i]
            if camera == 1:    # left camera
                y_steer[i] += delta_correction
            elif camera == 2:  # right camera
                y_steer[i] -= delta_correction
            else:              # central camera
                pass

        if augment_data:

            # mirror images with chance=0.5
            if random.choice([True, False]):
                X = X[:, :, ::-1, :]
                y_steer *= -1

            # perturb slightly steering direction
            for i in range(y_steer.shape[0]):
                y_steer[i] += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

            # if color images, randomly change brightness
            for i in range(X.shape[0]):
                # cv2.imshow('pre', np.uint8(cv2.resize(frame, (300, 150))))
                X[i] = cv2.cvtColor(X[i], code=cv2.COLOR_BGR2HSV)
                X[i][:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
                X[i][:, :, 2] = np.clip(X[i][:, :, 2], a_min=0, a_max=255)
                X[i] = cv2.cvtColor(X[i], code=cv2.COLOR_HSV2BGR)
                # cv2.imshow('post', np.uint8(cv2.resize(frame, (300, 150))))
                # cv2.waitKey()

        if normalize:
            # standardize features
            X = (X - np.mean(X)) / np.std(X)

        if K.backend() == 'theano':
            X = X.transpose(0, 3, 1, 2)

        yield X, y_steer


def get_model(summary=True):

    h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

    if K.backend() == 'theano':
        input_img = Input(shape=(c, h, w))
    else:
        input_img = Input(shape=(h, w, c))

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='valid', init='glorot_uniform')(input_img)
    x = Dropout(0.5)(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='valid', init='glorot_uniform')(input_img)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    out = Dense(output_dim=1)(x)

    model = Model(input=input_img, output=out)

    if summary:
        model.summary()

    return model


if __name__ == '__main__':

    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    x_batch, y_batch = load_data_batch(train_data)

    my_net = get_model(summary=True)

    opt = Adam(lr=1e-3)
    my_net.compile(optimizer=opt, loss='mse')

    # json dump of model architecture
    with open('logs/model.json', 'w') as f:
        f.write(my_net.to_json())

    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')

    my_net.fit_generator(generator=generate_data_batch(train_data),
                         samples_per_epoch=150*CONFIG['batchsize'],
                         nb_epoch=50,
                         validation_data=generate_data_batch(val_data, augment_data=False),
                         nb_val_samples=20*CONFIG['batchsize'],
                         callbacks=[checkpointer])
