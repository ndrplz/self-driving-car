from os.path import join
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import random
import keras.backend as K
from config import *
from load_data import load_data_batch, generate_data_batch, split_train_val


def get_nvidia_model(summary=True):

    if K.backend() == 'theano':
        input_frame = Input(shape=(CONFIG['input_channels'], NVIDIA_H, NVIDIA_W))
    else:
        input_frame = Input(shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels']))

    # input normalization
    x = Lambda(lambda z: z / 127.5 - 1.)(input_frame)

    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid')(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid')(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = ELU()(x)
    out = Dense(1)(x)

    model = Model(input=input_frame, output=out)

    if summary:
        model.summary()

    return model


if __name__ == '__main__':

    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    x_batch, y_batch = load_data_batch(train_data)

    nvidia_net = get_nvidia_model(summary=True)

    opt = Adam(lr=1e-3)
    nvidia_net.compile(optimizer=opt, loss='mse')

    # json dump of model architecture
    with open('logs/model.json', 'w') as f:
        f.write(nvidia_net.to_json())

    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/history.csv')

    nvidia_net.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
                         samples_per_epoch=300*CONFIG['batchsize'],
                         nb_epoch=50,
                         validation_data=generate_data_batch(val_data, augment_data=False),
                         nb_val_samples=100*CONFIG['batchsize'],
                         callbacks=[checkpointer, logger])
