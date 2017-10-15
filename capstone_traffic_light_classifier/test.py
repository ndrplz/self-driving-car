import cv2
import numpy as np
import tensorflow as tf
from traffic_light_dataset import TrafficLightDataset
from traffic_light_classifier import TrafficLightClassifier


checkpoint_file = './checkpoints/model_epoch_2.ckpt'


if __name__ == '__main__':

    # Parameters
    n_classes = 4                # Namely `void`, `red`, `yellow`, `green`
    input_h, input_w = 64, 64    # Shape to which input is resized

    # Init traffic light dataset
    dataset = TrafficLightDataset()
    dataset.init_from_npy('traffic_light_dataset.npy')

    # Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_h, input_w, 3])  # input placeholder
    p = tf.placeholder(dtype=tf.float32)                                     # dropout keep probability
    targets = tf.placeholder(dtype=tf.int32, shape=[None])

    # Define model
    classifier = TrafficLightClassifier(x, targets, p, n_classes, learning_rate=1e-4)

    # Add a saver to save the model after each epoch
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Restore pretrained weights
        saver.restore(sess, checkpoint_file)

        # Load a batch of data to test the model
        x_batch, y_batch = dataset.load_batch(batch_size=16)

        # Predict on loaded batch
        prediction = sess.run(fetches=classifier.inference, feed_dict={x: x_batch, targets: y_batch, p: 1.})
        prediction = np.argmax(prediction, axis=1)  # from onehot vectors to labels

        # Revert data normalization
        x_batch += np.abs(np.min(x_batch))
        x_batch *= 255
        x_batch = np.clip(x_batch, 0, 255).astype(np.uint8)

        # Qualitatively show results
        for b in range(x_batch.shape[0]):
            image = cv2.resize(x_batch[b], (256, 256))
            cv2.imshow('PRED {} GT {}'.format(prediction[b], y_batch[b]), image)
            cv2.waitKey()
