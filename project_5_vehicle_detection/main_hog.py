import os
import numpy as np
import pickle
from functions_detection import *
import scipy
from functions_utils import normalize_image
from functions_feat_extraction import find_cars
import time
import collections

from SSD import process_frame_bgr_with_SSD, get_SSD_model
ssd_model, bbox_helper, color_palette = get_SSD_model()

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


def prepare_output_blend_2(frame, img_detection, labeled_frame):

    h, w = frame.shape[:2]
    off_x, off_y = 20, 30
    thumb_h, thumb_w = (96, 128)

    vehicles_found = len(np.unique(labeled_frame)) - 1

    blend_out = frame.copy()

    # add a semi-transparent rectangle to highlight thumbnails on the left
    mask = cv2.rectangle(frame.copy(), (0, 0), ((2 + 1) * off_x + 2 * thumb_w, 2 * off_y + thumb_h), (0, 0, 0), thickness=cv2.FILLED)
    mask = cv2.rectangle(frame.copy(), (0, 0), (w, 2 * off_y + thumb_h), (0, 0, 0), thickness=cv2.FILLED)
    blend_out = cv2.addWeighted(src1=mask, alpha=0.3, src2=img_detection, beta=0.8, gamma=0)

    if vehicles_found > 0:

        vehicle_thumbnails = []
        for i in range(1, 1 + vehicles_found):
            y_nonzero, x_nonzero = np.where(labeled_frame == i)
            y_min_vehicle, y_max_vehicle = np.min(y_nonzero), np.max(y_nonzero)
            x_min_vehicle, x_max_vehicle = np.min(x_nonzero), np.max(x_nonzero)
            vehicle_thumb = frame[y_min_vehicle:y_max_vehicle, x_min_vehicle:x_max_vehicle, :]

            # draw car thumbnail on top-left
            vehicle_thumb = cv2.resize(vehicle_thumb, dsize=(thumb_w, thumb_h))
            start_x = 300 + i * off_x + (i - 1) * thumb_w  # should be i+1 and i, but i starts from 1
            blend_out[off_y:off_y+thumb_h, start_x:start_x+thumb_w, :] = vehicle_thumb

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_out, 'Vehicles in sight: {:02d}'.format(vehicles_found), (20, off_y+thumb_h//2), font, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.imshow('', frame)
    # cv2.waitKey()
    return blend_out


def process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False):

    hot_windows = []

    # SSD NET
    ssd_bboxes = process_frame_bgr_with_SSD(frame, ssd_model, bbox_helper, allow_classes=[7], min_confidence=0.3)
    for row in ssd_bboxes:
        label, confidence, x_min, y_min, x_max, y_max = row
        x_min = int(round(x_min * frame.shape[1]))
        y_min = int(round(y_min * frame.shape[0]))
        x_max = int(round(x_max * frame.shape[1]))
        y_max = int(round(y_max * frame.shape[0]))
        hot_windows.append(((x_min, y_min), (x_max, y_max)))

    # # HOG FEATS
    # for subsample in np.arange(1, 3, 0.5):
    #     hot_windows += find_cars(frame, 400, 600, subsample, svc, feature_scaler, feat_extraction_params)

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

    # img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)
    img_blend_out = prepare_output_blend_2(frame, img_detection, labeled_frame)

    if verbose:
        cv2.imshow('detection bboxes', img_hot_windows)
        cv2.imshow('heatmap', img_heatmap)
        cv2.imshow('labeled frame', img_labeling)
        cv2.imshow('detections', img_detection)
        cv2.waitKey()

    return img_blend_out


if __name__ == '__main__':

    # test_img_dir = 'test_images'
    # for test_img in os.listdir(test_img_dir):
    #
    #     t = time.time()
    #     print('Processing image {}...'.format(test_img), end="")
    #
    #     frame = cv2.imread(os.path.join(test_img_dir, test_img))
    #
    #     frame_out = process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=False, verbose=False)
    #
    #     cv2.imwrite('output_images/{}'.format(test_img), frame_out)
    #
    #     print('Done. Elapsed: {:.02f}'.format(time.time()-t))

    SSD_net, bbox_helper, color_palette = get_SSD_model()

    video_file = 'project_video.mp4'

    cap_in = cv2.VideoCapture(video_file)
    cap_out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1366, 720))

    while True:

        ret, frame = cap_in.read()

        if ret:

            frame_out = process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False)
            cap_out.write(frame_out)

            cv2.imshow('', frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap_in.release()
    cap_out.release()
    cv2.destroyAllWindows()
    exit()