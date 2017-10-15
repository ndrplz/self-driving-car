import cv2
import numpy as np
from glob import glob
from os.path import join


class TrafficLightDataset:

    def __init__(self):
        self.root = None
        self.initialized = False
        self.dataset_npy = None

    def init_from_files(self, dataset_root, resize=(256, 256)):
        """"
        Initialize the dataset from a certain dataset location on disk
        """
        self.root = dataset_root

        self.dataset_npy = []
        frame_list = glob(join(self.root, '*', '*.jpg'))

        print('Loading frames...')
        for frame_path in frame_list:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, resize[::-1])  # cv2 invert rows and cols
            label = self.infer_label_from_frame_path(frame_path)
            self.dataset_npy.append([frame, label])
        print('Done.')

        self.initialized = True

    def init_from_npy(self, dump_file_path):
        """
        Initialize the dataset from a previously created `.npy` dump
        """
        self.dataset_npy = np.load(dump_file_path)
        self.initialized = True

    def dump_to_npy(self, dump_file_path):
        """
        Dump the initialized dataset to a binary `.npy` file
        """
        if not self.initialized:
            raise IOError('Please initialize dataset first.')
        np.save(dump_file_path, self.dataset_npy)

    def load_batch(self, batch_size):

        if not self.initialized:
            raise IOError('Please initialize dataset first.')

        X_batch, Y_batch = [], []

        loaded = 0
        while loaded < batch_size:
            idx = np.random.randint(0, len(self.dataset_npy))
            x = self.dataset_npy[idx][0]
            y = self.dataset_npy[idx][1]

            X_batch.append(x)
            Y_batch.append(y)

            loaded += 1

        X_batch = self.preprocess(X_batch)

        return X_batch, Y_batch

    @staticmethod
    def preprocess(x):
        """
        Roughly center on zero and put in range [-1, 1]
        """
        x = np.float32(x) - np.mean(x)
        x /= x.max()
        return x

    @staticmethod
    def infer_label_from_frame_path(path):
        label = -1
        if 'none' in path:
            label = 0
        elif 'red' in path:
            label = 1
        elif 'yellow' in path:
            label = 2
        elif 'green' in path:
            label = 3
        return label

    def print_statistics(self):

        if not self.initialized:
            raise IOError('Please initialize dataset first.')

        color2label = {'none': 0, 'red': 1, 'yellow': 2, 'green': 3}

        statistics = {}
        for (color, num_label) in color2label.items():
            statistics[color] = np.sum(self.dataset_npy[:, 1] == num_label)
        print(statistics)
