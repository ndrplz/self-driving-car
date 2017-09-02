import sys
import os
from copy import deepcopy
from glob import glob
from unittest import mock

import numpy as np
import tensorflow as tf


def test_safe(func):
    """
    Isolate tests
    """
    def func_wrapper(*args):
        with tf.Graph().as_default():
            result = func(*args)
        print('Tests Passed')
        return result

    return func_wrapper


def _prevent_print(function, params):
    sys.stdout = open(os.devnull, "w")
    function(**params)
    sys.stdout = sys.__stdout__


def _assert_tensor_shape(tensor, shape, display_name):
    assert tf.assert_rank(tensor, len(shape), message='{} has wrong rank'.format(display_name))

    tensor_shape = tensor.get_shape().as_list() if len(shape) else []

    wrong_dimension = [ten_dim for ten_dim, cor_dim in zip(tensor_shape, shape)
                       if cor_dim is not None and ten_dim != cor_dim]
    assert not wrong_dimension, \
        '{} has wrong shape.  Found {}'.format(display_name, tensor_shape)


class TmpMock(object):
    """
    Mock a attribute.  Restore attribute when exiting scope.
    """
    def __init__(self, module, attrib_name):
        self.original_attrib = deepcopy(getattr(module, attrib_name))
        setattr(module, attrib_name, mock.MagicMock())
        self.module = module
        self.attrib_name = attrib_name

    def __enter__(self):
        return getattr(self.module, self.attrib_name)

    def __exit__(self, type, value, traceback):
        setattr(self.module, self.attrib_name, self.original_attrib)


@test_safe
def test_load_vgg(load_vgg, tf_module):
    with TmpMock(tf_module.saved_model.loader, 'load') as mock_load_model:
        vgg_path = ''
        sess = tf.Session()
        test_input_image = tf.placeholder(tf.float32, name='image_input')
        test_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        test_vgg_layer3_out = tf.placeholder(tf.float32, name='layer3_out')
        test_vgg_layer4_out = tf.placeholder(tf.float32, name='layer4_out')
        test_vgg_layer7_out = tf.placeholder(tf.float32, name='layer7_out')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        assert mock_load_model.called, \
            'tf.saved_model.loader.load() not called'
        assert mock_load_model.call_args == mock.call(sess, ['vgg16'], vgg_path), \
            'tf.saved_model.loader.load() called with wrong arguments.'

        assert input_image == test_input_image, 'input_image is the wrong object'
        assert keep_prob == test_keep_prob, 'keep_prob is the wrong object'
        assert vgg_layer3_out == test_vgg_layer3_out, 'layer3_out is the wrong object'
        assert vgg_layer4_out == test_vgg_layer4_out, 'layer4_out is the wrong object'
        assert vgg_layer7_out == test_vgg_layer7_out, 'layer7_out is the wrong object'


@test_safe
def test_layers(layers):
    num_classes = 2
    vgg_layer3_out = tf.placeholder(tf.float32, [None, None, None, 256])
    vgg_layer4_out = tf.placeholder(tf.float32, [None, None, None, 512])
    vgg_layer7_out = tf.placeholder(tf.float32, [None, None, None, 4096])
    layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    _assert_tensor_shape(layers_output, [None, None, None, num_classes], 'Layers Output')


@test_safe
def test_optimize(optimize):
    num_classes = 2
    shape = [2, 3, 4, num_classes]
    layers_output = tf.Variable(tf.zeros(shape))
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32)
    logits, train_op, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)

    _assert_tensor_shape(logits, [2*3*4, num_classes], 'Logits')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([train_op], {correct_label: np.arange(np.prod(shape)).reshape(shape), learning_rate: 10})
        test, loss = sess.run([layers_output, cross_entropy_loss], {correct_label: np.arange(np.prod(shape)).reshape(shape)})

    assert test.min() != 0 or test.max() != 0, 'Training operation not changing weights.'


@test_safe
def test_train_nn(train_nn):
    epochs = 1
    batch_size = 2

    def get_batches_fn(batach_size_parm):
        shape = [batach_size_parm, 2, 3, 3]
        return np.arange(np.prod(shape)).reshape(shape)

    train_op = tf.constant(0)
    cross_entropy_loss = tf.constant(10.11)
    input_image = tf.placeholder(tf.float32, name='input_image')
    correct_label = tf.placeholder(tf.float32, name='correct_label')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    with tf.Session() as sess:
        parameters = {
            'sess': sess,
            'epochs': epochs,
            'batch_size': batch_size,
            'get_batches_fn': get_batches_fn,
            'train_op': train_op,
            'cross_entropy_loss': cross_entropy_loss,
            'input_image': input_image,
            'correct_label': correct_label,
            'keep_prob': keep_prob,
            'learning_rate': learning_rate}
        _prevent_print(train_nn, parameters)


@test_safe
def test_for_kitti_dataset(data_dir):
    kitti_dataset_path = os.path.join(data_dir, 'data_road')
    training_labels_count = len(glob(os.path.join(kitti_dataset_path, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(kitti_dataset_path, 'training/image_2/*.png')))
    testing_images_count = len(glob(os.path.join(kitti_dataset_path, 'testing/image_2/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0),\
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(kitti_dataset_path)
    assert training_images_count == 289, 'Expected 289 training images, found {} images.'.format(training_images_count)
    assert training_labels_count == 289, 'Expected 289 training labels, found {} labels.'.format(training_labels_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)
