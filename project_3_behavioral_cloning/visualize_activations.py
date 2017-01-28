from load_data import preprocess
from os.path import join
import cv2
import numpy as np
import csv
from keras.models import Model
import matplotlib.pyplot as plt
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
    elu1_out_dir = 'activations_conv1_no_train'
    if not os.path.exists(elu1_out_dir):
        os.makedirs(elu1_out_dir)
    elu2_out_dir = 'activations_conv2_no_train'
    if not os.path.exists(elu2_out_dir):
        os.makedirs(elu2_out_dir)
    elu3_out_dir = 'activations_conv3_no_train'
    if not os.path.exists(elu3_out_dir):
        os.makedirs(elu3_out_dir)
    elu4_out_dir = 'activations_conv4_no_train'
    if not os.path.exists(elu4_out_dir):
        os.makedirs(elu4_out_dir)

    with open('data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    # load model architecture
    json_path = 'logs/model.json'
    model = model_from_json(open(json_path).read())

    # load model weights
    weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    first_ELU = Model(input=model.layers[0].input, output=model.layers[3].output)
    first_ELU.compile(optimizer='adam', loss='mse')

    second_ELU = Model(input=model.layers[0].input, output=model.layers[6].output)
    second_ELU.compile(optimizer='adam', loss='mse')

    third_ELU = Model(input=model.layers[0].input, output=model.layers[9].output)
    third_ELU.compile(optimizer='adam', loss='mse')

    fourth_ELU = Model(input=model.layers[0].input, output=model.layers[12].output)
    fourth_ELU.compile(optimizer='adam', loss='mse')

    for i, data_row in enumerate(driving_data):

        print('Frame {:06d} / {:06d}'.format(i, len(driving_data)))

        # ELU 1 ######################################################################

        plt.close('all')

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
        z = first_ELU.predict(central_frame)
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

        filename = join(elu1_out_dir, 'conv1_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')

        # ELU 2 ######################################################################

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
        z = second_ELU.predict(central_frame)
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

        filename = join(elu2_out_dir, 'conv2_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')

        # ELU 3 ####################################################

        plt.close('all')

        # load current color frame
        central_frame = cv2.imread(os.path.join('data', data_row[0]), cv2.IMREAD_COLOR)

        gs = gridspec.GridSpec(7, 8)
        ax = plt.subplot(gs[0, 3:5])
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(central_frame, cv2.COLOR_BGR2RGB))

        # preprocess and add batch dimension
        central_frame = preprocess(central_frame)
        central_frame = central_frame[np.newaxis, :, :, :]

        # z = np.random.rand(1, 14, 47, 36)
        z = third_ELU.predict(central_frame)
        z = normalize_in_0_255(z)
        rows, cols = 6, 8
        for r in range(rows):
            for c in range(cols):
                ax = plt.subplot(gs[r + 1, c])
                idx = r * cols + c
                cur_act = z[0, :, :, idx]
                cur_act = cv2.resize(cur_act, (NVIDIA_W, NVIDIA_H))
                ax.set_axis_off()
                ax.imshow(cur_act.astype(np.uint8), cmap='gray')
        plt.tight_layout(pad=0.1, h_pad=-10, w_pad=-1.5)

        filename = join(elu3_out_dir, 'conv3_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')

        # ELU 4 ####################################################

        plt.close('all')

        # load current color frame
        central_frame = cv2.imread(os.path.join('data', data_row[0]), cv2.IMREAD_COLOR)

        gs = gridspec.GridSpec(9, 8)
        ax = plt.subplot(gs[0, 3:5])
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(central_frame, cv2.COLOR_BGR2RGB))

        # preprocess and add batch dimension
        central_frame = preprocess(central_frame)
        central_frame = central_frame[np.newaxis, :, :, :]

        # z = np.random.rand(1, 14, 47, 36)
        z = fourth_ELU.predict(central_frame)
        z = normalize_in_0_255(z)
        rows, cols = 8, 8
        for r in range(rows):
            for c in range(cols):
                ax = plt.subplot(gs[r + 1, c])
                idx = r * cols + c
                cur_act = z[0, :, :, idx]
                cur_act = cv2.resize(cur_act, (NVIDIA_W, NVIDIA_H))
                ax.set_axis_off()
                ax.imshow(cur_act.astype(np.uint8), cmap='gray')
        plt.tight_layout(pad=0.4, h_pad=-2, w_pad=-1.5)

        filename = join(elu4_out_dir, 'conv4_{:06d}.jpg'.format(i))
        plt.savefig(filename, facecolor='black', bbox_inches='tight')
