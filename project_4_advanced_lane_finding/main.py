import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize
from perspective_utils import birdeye
from line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
from moviepy.editor import VideoFileClip
import numpy as np

processed_frames = 0
line_lt, line_rt = Line(buffer_len=5), Line(buffer_len=5)


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt):

    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    mean_curvature = np.mean([line_lt.curvature, line_rt.curvature])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}'.format(mean_curvature), (860, 40), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def process_pipeline(frame, keep_state=True):

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, img_birdeye, Minv, line_lt, line_rt, keep_state)

    dewarped = cv2.warpPerspective(img_fit, Minv, img_fit.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    idx = np.any([dewarped != 0][0], axis=2)
    mask = blend_on_road.copy()
    mask[idx] = dewarped[idx]
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.5, src2=blend_on_road, beta=0.5, gamma=0.)

    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt)

    processed_frames += 1

    return blend_output


if __name__ == '__main__':

    # first things first: calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    selector = 'project'
    clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
    clip.write_videofile('out_{}.mp4'.format(selector), audio=False)

    # for test_img in glob.glob('test_images2/*.jpg'):
    #
    #     frame = cv2.imread(test_img)
    #
    #     blend = process_pipeline(frame, keep_state=False)
    #
    #     plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
    #
    #     plt.show()
