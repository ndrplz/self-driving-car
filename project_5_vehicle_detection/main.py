import os
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
from functions_detection import *
from scipy.ndimage.measurements import label
from functions_utils import normalize_image


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

    # compute windows to classify
    windows_multiscale = compute_windows_multiscale(frame, verbose=False)

    # classify each window
    hot_windows = search_windows(frame, windows_multiscale, svc, feature_scaler, feat_extraction_params)

    # compute heatmaps positive windows found
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, verbose=False)

    # label connected components
    labeled_frame, num_objects = label(heatmap_thresh)

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

    mode = 'images'

    # load pretrained svm classifier
    svc = pickle.load(open('data/svm_trained.pickle', 'rb'))

    # load feature scaler fitted on training data
    scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

    # load parameters used to perform feature extraction
    feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))

    if mode == 'video':
        time_window = 1
        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(process_pipeline)
        clip.write_videofile('out_{}_{}.mp4'.format(selector, time_window), audio=False)

    else:

        test_img_dir = 'test_images'
        for test_img in os.listdir(test_img_dir):

            frame = cv2.imread(os.path.join(test_img_dir, test_img))

            frame_out = process_pipeline(frame, svc, scaler, feat_extraction_params, keep_state=False, verbose=False)

            cv2.imwrite('output_images/{}'.format(test_img), frame_out)
