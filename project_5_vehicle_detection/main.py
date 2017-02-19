import os
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from functions_detection import *
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt


def process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False):

    # compute windows to classify
    windows_multiscale = compute_windows_multiscale(frame, verbose=False)

    # classify each window
    hot_windows = search_windows(frame, windows_multiscale, svc, feature_scaler, feat_extraction_params)

    # compute heatmaps positive windows found
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, verbose=False)

    # label connected components
    labeled_frame, num_objects = label(heatmap_thresh)

    # draw bounding boxes
    frame_out = draw_labeled_bounding_boxes(frame, labeled_frame, num_objects)

    if verbose:
        # draw `hot boxes` on current frame
        window_img = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=1)
        plt.imshow(window_img)
        plt.imshow(labeled_frame, cmap='Accent')
        plt.imshow(cv2.cvtColor(frame_out, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_out


if __name__ == '__main__':

    mode = 'images'

    # load pretrained svm classifier
    svc = pickle.load(open('data/svm_trained.pickle', 'rb'))

    # load feature scaler fitted on training data
    scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

    # load parameters used to perform feature extraction
    feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))

    if mode == 'video':
        pass
        # selector = 'project'
        # clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
        # clip.write_videofile('out_{}_{}.mp4'.format(selector, time_window), audio=False)

    else:

        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):

            frame = cv2.imread(os.path.join(test_img_dir, test_img))

            frame_out = process_pipeline(frame, svc, scaler, feat_extraction_params, keep_state=False, verbose=False)

            cv2.imwrite('output_images/{}'.format(test_img), frame_out)
