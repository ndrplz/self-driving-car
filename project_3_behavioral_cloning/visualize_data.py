from load_data import load_data_batch, split_train_val
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as path


def visualize_steering_distribution(train_data):
    """
    Visualize the training ground truth distribution "as provided"
    :param train_data: list of udacity training data
    :return:
    """
    train_steering = np.float32(np.array(train_data)[:, 3])
    plt.title('Steering angle distribution in training data')
    plt.hist(train_steering, 100, normed=0, facecolor='green', alpha=0.75)
    plt.ylabel('# frames'), plt.xlabel('steering angle')
    plt.show()


def visualize_bias_parameter_effect(train_data):
    """
    Visualize how the 'bias' parameter influences the ground truth distribution
    :param train_data:
    :return:
    """
    biases = np.linspace(start=0., stop=1., num=5)
    fig, axarray = plt.subplots(len(biases))
    plt.suptitle('Effect of bias parameter on steering angle distribution', fontsize=14, fontweight='bold')
    for i, ax in enumerate(axarray.ravel()):
        b = biases[i]
        x_batch, y_batch = load_data_batch(train_data, batchsize=1024, augment_data=True, bias=b)
        ax.hist(y_batch, 50, normed=1, facecolor='green', alpha=0.75)
        ax.set_title('Bias: {:02f}'.format(b))
        ax.axis([-1., 1., 0., 2.])
    plt.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)
    plt.show()




if __name__ == '__main__':

    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

    # visualize_steering_distribution(train_data)

    # visualize_bias_parameter_effect(train_data)

    # np.float(train_data[:, 3])

    from load_data import preprocess

    for i in range(len(train_data)):
        central_frame = cv2.imread(path.join('data', train_data[i][0]))
        steering = np.float32(train_data[i][3])
        if train_data[i][0] == 'IMG/center_2016_12_01_13_34_43_116.jpg'\
            or train_data[i][0] == 'IMG/center_2016_12_01_13_46_20_434.jpg'\
                or train_data[i][0] == 'IMG/center_2016_12_01_13_39_48_531.jpg':

            plt.close('all')
            proc_frame = preprocess(central_frame)
            plt.imshow(cv2.cvtColor(proc_frame.astype(np.uint8), code=cv2.COLOR_BGR2RGB))
            print('{}'.format(train_data[i][0]))
            plt.title('Steering: {:03f}'.format(steering))
            plt.gca().set_axis_off()
            filename = path.join('.', '{}.png'.format(train_data[i][0]))
            plt.savefig(filename, facecolor='white', bbox_inches='tight')

            plt.close('all')
            plt.imshow(cv2.cvtColor(central_frame, code=cv2.COLOR_BGR2RGB))
            print('{}'.format(train_data[i][0]))
            plt.title('Steering: {:03f}'.format(steering))
            plt.gca().set_axis_off()
            filename = path.join('.', '{}'.format(train_data[i][0]))
            plt.savefig(filename, facecolor='white', bbox_inches='tight')








