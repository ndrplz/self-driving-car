import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


white_HSV_th_min = np.array([20, 0, 180])
white_HSV_th_max = np.array([255, 80, 255])

yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])


def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min_th_ok = np.all(HSV > min_values, axis=2)
    max_th_ok = np.all(HSV < max_values, axis=2)

    out = np.logical_and(min_th_ok, max_th_ok)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()

    return out


def thresh_frame_sobel(frame, kernel_size):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

    return sobel_mag.astype(bool)


def get_binary_from_equalized_grayscale(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    return th


def binarize(img, verbose=False):

    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    HSV_white_mask = thresh_frame_in_HSV(img, white_HSV_th_min, white_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_white_mask)

    HSV_yellow_mask = thresh_frame_in_HSV(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_yellow_mask)

    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask)

    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        f, ax = plt.subplots(3, 3)
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 2].imshow(sobel_mask, cmap='gray')
        ax[0, 2].set_title('sobel binary')
        ax[0, 2].set_axis_off()
        ax[1, 0].imshow(HSV_white_mask, cmap='gray')
        ax[1, 0].set_title('white binary')
        ax[1, 0].set_axis_off()
        ax[1, 1].imshow(HSV_yellow_mask, cmap='gray')
        ax[1, 1].set_title('yellow binary')
        ax[1, 1].set_axis_off()
        ax[1, 2].imshow(binary, cmap='gray')
        ax[1, 2].set_title('before close')
        ax[1, 2].set_axis_off()
        ax[2, 0].imshow(eq_white_mask, cmap='gray')
        ax[2, 0].set_title('equalization')
        ax[2, 0].set_axis_off()
        ax[2, 2].imshow(closing, cmap='gray')
        ax[2, 2].set_title('after close')
        ax[2, 2].set_axis_off()
        plt.show()

    return closing


if __name__ == '__main__':

    test_images = glob.glob('test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binarize(img=img, verbose=True)
