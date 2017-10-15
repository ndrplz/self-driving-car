import tensorflow as tf
from traffic_light_dataset import TrafficLightDataset
from traffic_light_classifier import TrafficLightClassifier


if __name__ == '__main__':

    # Parameters
    n_classes = 4                # Namely `void`, `red`, `yellow`, `green`
    input_h, input_w = 64, 64    # Shape to which input is resized

    # Init traffic light dataset
    dataset = TrafficLightDataset()
    # dataset_root = 'C:/Users/minotauro/Desktop/traffic_light_dataset'
    # dataset.init_from_files(dataset_root, resize=(input_h, input_w))
    # dataset.dump_to_npy('traffic_light_dataset.npy')
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
                                              feed_dict={x: x_batch, targets: y_batch, p: 0.5})

                loss_cur_epoch += loss_this_batch

            loss_cur_epoch /= batches_each_epoch
            print('Loss cur epoch: {:.04f}'.format(loss_cur_epoch))

            # Eventually evaluate on whole test set when training ends
            average_test_accuracy = 0.0
            num_test_batches = 500
            for _ in range(num_test_batches):
                x_batch, y_batch = dataset.load_batch(batch_size)
                average_test_accuracy += sess.run(fetches=classifier.accuracy,
                                                  feed_dict={x: x_batch, targets: y_batch, p: 1.})
            average_test_accuracy /= num_test_batches
            print('Training accuracy: {:.03f}'.format(average_test_accuracy))
            print('*' * 50)

            # Save the variables to disk.
            save_path = saver.save(sess, './checkpoints/model_epoch_{}.ckpt'.format(epoch))

            epoch += 1
