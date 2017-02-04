import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road


if __name__ == '__main__':

    # first things first: calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        # undistort the image using coefficients found in calibration
        img_undistorted = undistort(img, mtx, dist, verbose=False)

        # binarize the frame s.t. lane lines are highlighted as much as possible
        img_binary = binarize(img_undistorted, verbose=False)

        # compute perspective transform to obtain bird's eye view
        img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

        # fit 2-degree polynomial curve onto lane lines found
        left_fit, right_fit = get_fits_by_sliding_windows(img_birdeye, n_windows=9, verbose=False)

        # draw the surface enclosed by lane lines back onto the original frame
        blend_on_road = draw_back_onto_the_road(img_undistorted, img_birdeye, Minv, left_fit, right_fit)

        plt.imshow(cv2.cvtColor(blend_on_road, code=cv2.COLOR_BGR2RGB))
        plt.show()

