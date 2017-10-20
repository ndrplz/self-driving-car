import tensorflow as tf
from os import makedirs
from os.path import exists, join
from traffic_light_dataset import TrafficLightDataset
from traffic_light_classifier import TrafficLightClassifier


if __name__ == '__main__':

    # Parameters
    input_h, input_w = 128, 128    # Shape to which input is resized

    # Init traffic light dataset
    dataset = TrafficLightDataset()
    dataset_file = 'traffic_light_dataset_npy/traffic_light_dataset_mixed_resize_{}.npy'.format(input_h)
    dataset.init_from_npy(dataset_file)

    # Define model
    classifier = TrafficLightClassifier(input_shape=[input_h, input_w], learning_rate=1e-4, verbose=True)

    # Checkpoint stuff
    saver = tf.train.Saver()  # saver to save the model after each epoch
    checkpoint_dir = './checkpoint_mixed_{}'.format(input_h)  # checkpoint directory
    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Training parameters
        batch_size         = 32
        batches_each_epoch = 1000

        epoch = 0

        while True:

            loss_cur_epoch = 0

            for _ in range(batches_each_epoch):

                # Load a batch of training data
                x_batch, y_batch = dataset.load_batch(batch_size, augmentation=True)

                # Actually run one training step here
                _, loss_this_batch = sess.run(fetches=[classifier.train_step, classifier.loss],
                                              feed_dict={classifier.x: x_batch,
                                                         classifier.targets: y_batch,
                                                         classifier.keep_prob: 0.5})

                loss_cur_epoch += loss_this_batch

            loss_cur_epoch /= batches_each_epoch
            print('Loss cur epoch: {:.04f}'.format(loss_cur_epoch))

            # Eventually evaluate on whole test set when training ends
            average_test_accuracy = 0.0
            num_test_batches = 500
            for _ in range(num_test_batches):
                x_batch, y_batch = dataset.load_batch(batch_size)
                average_test_accuracy += sess.run(fetches=classifier.accuracy,
                                                  feed_dict={classifier.x: x_batch,
                                                             classifier.targets: y_batch,
                                                             classifier.keep_prob: 1.0})
            average_test_accuracy /= num_test_batches
            print('Training accuracy: {:.03f}'.format(average_test_accuracy))
            print('*' * 50)

            # Save the variables to disk.
            save_path = saver.save(sess, join(checkpoint_dir, 'TLC_epoch_{}.ckpt'.format(epoch)))

            epoch += 1
