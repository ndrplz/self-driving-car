import numpy as np
import tensorflow as tf


EPS  = np.finfo('float32').eps


class TrafficLightClassifier:

    def __init__(self, input_shape, learning_rate, verbose=True):

        # Placeholders
        input_h, input_w = input_shape
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_h, input_w, 3])  # input placeholder
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32)  # dropout keep probability

        self.n_classes      = 4              # {void, red, yellow, green}
        self.learning_rate  = learning_rate  # learning rate used in train step

        self._inference     = None
        self._loss          = None
        self._train_step    = None
        self._accuracy      = None
        self._summaries     = None

        self.inference
        self.loss
        self.train_step
        self.accuracy
        # self.summaries # todo add these

        if verbose:
            self.print_summary()

    @property
    def inference(self):
        if self._inference is None:
            with tf.variable_scope('inference'):

                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

                conv1_filters = 32
                conv1 = tf.layers.conv2d(self.x, conv1_filters, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

                conv2_filters = 64
                conv2 = tf.layers.conv2d(pool1, conv2_filters, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

                _, h, w, c = pool2.get_shape().as_list()
                pool2_flat = tf.reshape(pool2, shape=[-1, h * w * c])

                pool2_drop = tf.nn.dropout(pool2_flat, keep_prob=self.keep_prob)

                hidden_units = self.n_classes
                hidden = tf.layers.dense(pool2_drop, units=hidden_units, activation=tf.nn.relu)

                logits = tf.layers.dense(hidden, units=self.n_classes, activation=None)

                self._inference = tf.nn.softmax(logits)

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            with tf.variable_scope('loss'):
                predictions = self.inference
                targets_onehot = tf.one_hot(self.targets, depth=self.n_classes)
                self._loss = tf.reduce_mean(-tf.reduce_sum(targets_onehot * tf.log(predictions + EPS), reduction_indices=1))
        return self._loss

    @property
    def train_step(self):
        if self._train_step is None:
            with tf.variable_scope('training'):
                self._train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        return self._train_step

    @property
    def accuracy(self):
        if self._accuracy is None:
            with tf.variable_scope('accuracy'):
                correct_predictions = tf.equal(tf.argmax(self.inference, axis=1),
                                               tf.argmax(tf.one_hot(self.targets, depth=self.n_classes), axis=1))
                self._accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return self._accuracy

    @staticmethod
    def print_summary():
        def pretty_border():
            print('*' * 50)

        pretty_border()
        print('Classifier initialized.')

        trainable_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        num_trainable_params = np.sum([np.prod(v.get_shape()) for v in trainable_variables])
        print('Number of trainable parameters: {}'.format(num_trainable_params))
        pretty_border()
