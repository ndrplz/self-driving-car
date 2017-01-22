from load_data import load_data_batch, split_train_val, preprocess
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
from keras.models import model_from_json
import os
import matplotlib.gridspec as gridspec


def normalize_in_0_255(tensor):
    tensor -= tensor.min()
    tensor /= tensor.max()
    tensor *= 255.
    return tensor


if __name__ == '__main__':

    # directory in which activations are saved
    conv1_out_dir = 'activations_conv1'
    if not os.path.exists(conv1_out_dir):
        os.makedirs(conv1_out_dir)
    conv2_out_dir = 'activations_conv2'
    if not os.path.exists(conv2_out_dir):
        os.makedirs(conv2_out_dir)

    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    # load model architecture
    json_path = 'logs/model.json'
    model = model_from_json(open(json_path).read())

    # load model weights
    weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    first_conv = Model(input=model.layers[0].input, output=model.layers[2].output)
    first_conv.compile(optimizer='adam', loss='mse')

    second_conv = Model(input=model.layers[0].input, output=model.layers[5].output)
    second_conv.compile(optimizer='adam', loss='mse')

    for i, data_row in enumerate(train_data):

        print('Frame {:06d} / {:06d}'.format(i, len(train_data)))

        plt.close('all')

        # CONV1 ######################################################################

        # load current color frame
        central_frame = cv2.imread(os.path.join('data', data_row[0]), cv2.IMREAD_COLOR)

        gs = gridspec.GridSpec(4, 8)
        ax = plt.subplot(gs[0, 3:5])
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(central_frame, cv2.COLOR_BGR2RGB))

        # preprocess and add batch dimension
        central_frame = preprocess(central_frame)
        central_frame = central_frame[np.newaxis, :, :, :]

        # z = np.random.rand(1, 31, 98, 24)
        z = first_conv.predict(central_frame)
        z = normalize_in_0_255(z)
        rows, cols = 3, 8
        for r in range(rows):
            for c in range(cols):
                ax = plt.subplot(gs[r+1, c])
                idx = r*cols + c
                cur_act = z[0, :, :, idx]
                cur_act = cv2.resize(cur_act, (NVIDIA_W, NVIDIA_H))
                ax.set_axis_off()
                ax.imshow(cur_act.astype(np.uint8), cmap='gray')
        plt.tight_layout(pad=0.1, h_pad=-10, w_pad=-1.5)

        filename = join(conv1_out_dir, 'conv1_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')


        # CONV2 ######################################################################

        plt.close('all')

        # load current color frame
        central_frame = cv2.imread(os.path.join('data', data_row[0]), cv2.IMREAD_COLOR)

        gs = gridspec.GridSpec(5, 9)
        ax = plt.subplot(gs[0, 3:6])
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(central_frame, cv2.COLOR_BGR2RGB))

        # preprocess and add batch dimension
        central_frame = preprocess(central_frame)
        central_frame = central_frame[np.newaxis, :, :, :]

        # z = np.random.rand(1, 14, 47, 36)
        z = second_conv.predict(central_frame)
        z = normalize_in_0_255(z)
        rows, cols = 4, 9
        for r in range(rows):
            for c in range(cols):
                ax = plt.subplot(gs[r + 1, c])
                idx = r * cols + c
                cur_act = z[0, :, :, idx]
                cur_act = cv2.resize(cur_act, (NVIDIA_W, NVIDIA_H))
                ax.set_axis_off()
                ax.imshow(cur_act.astype(np.uint8), cmap='gray')
        plt.tight_layout(pad=0.1, h_pad=-10, w_pad=-1.5)

        filename = join(conv2_out_dir, 'conv2_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')



