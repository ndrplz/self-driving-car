"""
Dirty and running file to use Python2.7

Dependency form helper and unittests have been removed due to compatibility issues.

Once training is done, code will be moved to `main.py`
"""
from __future__ import division
import tensorflow as tf
import warnings
from distutils.version import LooseVersion
from os.path import join, expanduser
import re
import random
import shutil
import numpy as np
import os.path
import scipy.misc
import time
from glob import glob


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                h, w = gt_bg.shape
                gt_bg = gt_bg.reshape(h, w, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.

    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    For reference: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    kernel_regularizer = tf.contrib.layers.l2_regularizer(0.5)

    # Compute logits
    layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)
    layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=[1, 1],
                                     padding='same', kernel_regularizer=kernel_regularizer)

    # Add skip connection before 4th and 7th layer
    layer7_logits_up   = tf.image.resize_images(layer7_logits, size=[10, 36])
    layer_4_7_fused = tf.add(layer7_logits_up, layer4_logits)

    # Add skip connection before (4+7)th and 3rd layer
    layer_4_7_fused_up = tf.image.resize_images(layer_4_7_fused, size=[20, 72])
    layer_3_4_7_fused = tf.add(layer3_logits, layer_4_7_fused_up)

    # resize to original size
    layer_3_4_7_up = tf.image.resize_images(layer_3_4_7_fused, size=[160, 576])

    return layer_3_4_7_up


def optimize(net_prediction, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param net_prediction: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Unroll
    logits_flat = tf.reshape(net_prediction, (-1, num_classes))
    labels_flat = tf.reshape(labels, (-1, num_classes))

    # Define loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=logits_flat))

    # Define optimization step
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits_flat, train_step, cross_entropy_loss


def train_nn(sess, training_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
             image_input, labels, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param training_epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Variable initialization
    sess.run(tf.global_variables_initializer())

    lr = 1e-4
    examples_each_epoch = 100

    for e in range(0, training_epochs):

        loss_this_epoch = 0.0

        for i in range(0, examples_each_epoch):

            # Load a batch of examples
            batch_x, batch_y = next(get_batches_fn(batch_size))

            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={image_input: batch_x, labels: batch_y, keep_prob: 0.25, learning_rate: lr})

            loss_this_epoch += cur_loss

        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(e, loss_this_epoch / examples_each_epoch))


def run():

    num_classes = 2

    image_h, image_w = (160, 576)

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = join(data_dir, 'vgg')

        # Create function to get batches
        batch_generator = gen_batch_function(join(data_dir, 'data_road/training'), (image_h, image_w))

        # Load VGG pretrained
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # Add skip connections
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # Define placeholders
        labels = tf.placeholder(tf.float32, shape=[None, image_h, image_w, num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        logits, train_op, cross_entropy_loss = optimize(output, labels, learning_rate, num_classes)

        # Training parameters
        training_epochs = 40
        batch_size      = 8

        train_nn(sess, training_epochs, batch_size, batch_generator, train_op, cross_entropy_loss,
                 image_input, labels, keep_prob, learning_rate)

        save_inference_samples(runs_dir, data_dir, sess, (image_h, image_w), logits, keep_prob, image_input)


if __name__ == '__main__':

    data_dir = join(expanduser("~"), 'code', 'self-driving-car', 'project_12_road_segmentation', 'data')
    runs_dir = join(expanduser("~"), 'majinbu_home', 'road_segmentation_prediction')
    # runs_dir = join(expanduser("~"), 'code', 'self-driving-car', 'project_12_road_segmentation', 'runs')

    run()
