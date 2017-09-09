import os
import argparse
import warnings
import tensorflow as tf
from helper import gen_batch_function, save_inference_samples
from distutils.version import LooseVersion
from os.path import join, expanduser
import project_tests as tests
from image_augmentation import perform_augmentation


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'),\
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
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
    layer7_logits_up = tf.image.resize_images(layer7_logits, size=[10, 36])
    layer_4_7_fused = tf.add(layer7_logits_up, layer4_logits)

    # Add skip connection before (4+7)th and 3rd layer
    layer_4_7_fused_up = tf.image.resize_images(layer_4_7_fused, size=[20, 72])
    layer_3_4_7_fused = tf.add(layer3_logits, layer_4_7_fused_up)

    # resize to original size
    layer_3_4_7_up = tf.image.resize_images(layer_3_4_7_fused, size=[160, 576])
    layer_3_4_7_up = tf.layers.conv2d(layer_3_4_7_up, num_classes, kernel_size=[15, 15],
                                      padding='same', kernel_regularizer=kernel_regularizer)

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

    lr = args.learning_rate

    for e in range(0, training_epochs):

        loss_this_epoch = 0.0

        for i in range(0, args.batches_per_epoch):

            # Load a batch of examples
            batch_x, batch_y = next(get_batches_fn(batch_size))
            if should_do_augmentation:
                batch_x, batch_y = perform_augmentation(batch_x, batch_y)

            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={image_input: batch_x, labels: batch_y, keep_prob: 0.25,
                                              learning_rate: lr})

            loss_this_epoch += cur_loss

        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(e, loss_this_epoch / args.batches_per_epoch))


def perform_tests():
    tests.test_for_kitti_dataset(data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)


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
        train_nn(sess, args.training_epochs, args.batch_size, batch_generator, train_op, cross_entropy_loss,
                 image_input, labels, keep_prob, learning_rate)

        save_inference_samples(runs_dir, data_dir, sess, (image_h, image_w), logits, keep_prob, image_input)


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size used for training', metavar='')
    parser.add_argument('--batches_per_epoch', type=int, default=100, help='Batches each training epoch', metavar='')
    parser.add_argument('--training_epochs', type=int, default=30, help='Number of training epoch', metavar='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate', metavar='')
    parser.add_argument('--augmentation', type=bool, default=True, help='Perform augmentation in training', metavar='')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use', metavar='')
    return parser.parse_args()


if __name__ == '__main__':

    data_dir = join(expanduser("~"), 'code', 'self-driving-car', 'project_12_road_segmentation', 'data')
    runs_dir = join(expanduser("~"), 'majinbu_home', 'road_segmentation_prediction')

    args = parse_arguments()

    # Appropriately set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('Using GPU: {:02d}.'.format(args.gpu))

    # Turn off augmentation during tests
    should_do_augmentation = False
    perform_tests()

    # Restore appropriate augmentation value
    should_do_augmentation = args.augmentation
    run()
