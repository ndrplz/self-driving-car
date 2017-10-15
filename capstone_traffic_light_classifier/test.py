import argparse
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join
from traffic_light_dataset import TrafficLightDataset
from traffic_light_classifier import TrafficLightClassifier


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--mode', type=str, choices=['from_npy', 'from_file', 'from_dir'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resize_h', type=int, default=64, help='Height to which input is resized')
    parser.add_argument('--resize_w', type=int, default=64, help='Width to which input is resized')
    return parser.parse_args()


def load_test_data(args):
    """
    Load a data batch to perform inference, according to selected mode.
    """
    dataset = TrafficLightDataset()

    def read_and_resize_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return cv2.resize(image, (args.resize_h, args.resize_w))

    # Load a random batch of data from a `.npy` dataset
    if args.mode == 'from_npy':
        dataset.init_from_npy(args.data_path)                 # Init traffic light dataset
        x_batch, y_batch = dataset.load_batch(batch_size=16)  # Random batch of examples

    # Load a single image from disk
    elif args.mode == 'from_file':
        image = read_and_resize_image(args.data_path)
        x_batch = np.expand_dims(dataset.preprocess(image), 0)
        y_batch = np.ones(shape=(x_batch.shape[0], 1)) * -1   # -1 means label not available

    # Load all images in a certain directory
    elif args.mode == 'from_dir':
        x_batch, y_batch = [], []
        image_list = glob(join(args.data_path, '*.jpg'))
        for image_path in image_list:
            image = read_and_resize_image(image_path)
            x_batch.append(dataset.preprocess(image))
            y_batch.append([-1])                              # -1 means label not available
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
    else:
        raise ValueError('Mode: "{}" not supported.'.format(args.mode))

    return x_batch, y_batch


if __name__ == '__main__':

    # Parse command line arguments
    args = parse_arguments()

    # Load data on which prediction will be performed
    x_batch, y_batch = load_test_data(args)

    # Define model
    classifier = TrafficLightClassifier(input_shape=[args.resize_h, args.resize_w], learning_rate=1e-4)

    # Add a saver to save the model after each epoch
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore pretrained weights
        saver.restore(sess, args.checkpoint_path)

        # Predict on loaded batch
        prediction = sess.run(fetches=classifier.inference,
                              feed_dict={classifier.x: x_batch, classifier.keep_prob: 1.})
        prediction = np.argmax(prediction, axis=1)  # from onehot vectors to labels

        # Qualitatively show results
        for b in range(x_batch.shape[0]):

            # Revert data normalization
            image = x_batch[b]
            image += np.abs(np.min(image))
            image *= 255
            image = np.clip(image, 0, 255).astype(np.uint8)

            # Display result
            image = cv2.resize(image, (256, 256))
            cv2.imshow('PRED {} GT {}'.format(prediction[b], y_batch[b]), image)
            cv2.waitKey()
