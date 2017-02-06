import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line
from moviepy.editor import VideoFileClip
import numpy as np


line_lt, line_rt = Line(buffer_len=5), Line(buffer_len=5)


def process_pipeline(frame):

    global line_lt, line_rt

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, img_birdeye, Minv, line_lt.average_fit, line_rt.average_fit)

    thumb_binary = cv2.resize(img_binary, dsize=None, fx=0.2, fy=0.2)
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary])
    blend_on_road[0:144, 0:256, :] = thumb_binary

    thumb_birdeye = cv2.resize(img_birdeye, dsize=None, fx=0.2, fy=0.2)
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[0:144, 300:556, :] = thumb_birdeye

    thumb_img_fit = cv2.resize(img_fit, dsize=None, fx=0.2, fy=0.2)
    blend_on_road[0:144, 600:856, :] = thumb_img_fit

    return blend_on_road


if __name__ == '__main__':

    # first things first: calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    selector = 'project'
    clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
    clip.write_videofile('out_{}.mp4'.format(selector), audio=False)

    # for test_img in glob.glob('test_images/*.jpg'):
    #
    #     frame = cv2.imread(test_img)
    #
    #     blend = process_pipeline(frame)
    #
    #     plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
    #
    #     plt.show()
