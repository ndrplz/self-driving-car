import os
import numpy as np
import pickle
from functions_detection import *
import scipy
from functions_utils import normalize_image
from functions_feat_extraction import find_cars
import time
import collections

time_window = 5
hot_windows_history = collections.deque(maxlen=time_window)

# load pretrained svm classifier
svc = pickle.load(open('data/svm_trained.pickle', 'rb'))

# load feature scaler fitted on training data
feature_scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

# load parameters used to perform feature extraction
feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))


def prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection):

    h, w, c = frame.shape

    # decide the size of thumbnail images
    thumb_ratio = 0.25
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    # resize to thumbnails images from various stages of the pipeline
    thumb_hot_windows = cv2.resize(img_hot_windows, dsize=(thumb_w, thumb_h))
    thumb_heatmap = cv2.resize(img_heatmap, dsize=(thumb_w, thumb_h))
    thumb_labeling = cv2.resize(img_labeling, dsize=(thumb_w, thumb_h))

    off_x, off_y = 20, 45

    # add a semi-transparent rectangle to highlight thumbnails on the left
    mask = cv2.rectangle(img_detection.copy(), (0, 0), (2*off_x + thumb_w, h), (0, 0, 0), thickness=cv2.FILLED)
    img_blend = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_detection, beta=0.8, gamma=0)

    # stitch thumbnails
    img_blend[off_y:off_y+thumb_h, off_x:off_x+thumb_w, :] = thumb_hot_windows
    img_blend[2*off_y+thumb_h:2*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_heatmap
    img_blend[3*off_y+2*thumb_h:3*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_labeling

    return img_blend


def process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False):

    hot_windows = []

    for subsample in np.arange(1, 3, 0.5):
        hot_windows += find_cars(frame, 400, 600, subsample, svc, feature_scaler, feat_extraction_params)

    if keep_state:
        if hot_windows:
            hot_windows_history.append(hot_windows)
            hot_windows = np.concatenate(hot_windows_history)

    # compute heatmaps positive windows found
    thresh = (time_window - 1) if keep_state else 0
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=thresh, verbose=False)

    # label connected components
    labeled_frame, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)

    # prepare images for blend
    img_hot_windows = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=2)                 # show pos windows
    img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)         # draw heatmap
    img_labeling = cv2.applyColorMap(normalize_image(labeled_frame), colormap=cv2.COLORMAP_HOT)  # draw label
    img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)        # draw detected bboxes

    img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)

    if verbose:
        cv2.imshow('detection bboxes', img_hot_windows)
        cv2.imshow('heatmap', img_heatmap)
        cv2.imshow('labeled frame', img_labeling)
        cv2.imshow('detections', img_detection)
        cv2.waitKey()

    return img_blend_out


if __name__ == '__main__':

    test_img_dir = 'test_images'
    for test_img in os.listdir(test_img_dir):

        t = time.time()
        print('Processing image {}...'.format(test_img), end="")

        frame = cv2.imread(os.path.join(test_img_dir, test_img))

        frame_out = process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=False, verbose=False)

        cv2.imwrite('output_images/{}'.format(test_img), frame_out)

        print('Done. Elapsed: {:.02f}'.format(time.time()-t))

