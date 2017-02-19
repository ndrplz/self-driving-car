import numpy as np
import cv2
from skimage.feature import hog
from computer_vision_utils.stitching import stitch_together


def compute_windows_multiscale(image, verbose=False):

    h, w, c = image.shape

    windows_multiscale = []

    windows_32 = slide_window(image, x_start_stop=[None, None], y_start_stop=[4 * h//8, 5 * h//8],
                                   xy_window=(32, 32), xy_overlap=(0.8, 0.8))
    windows_multiscale.append(windows_32)

    windows_64 = slide_window(image, x_start_stop=[None, None], y_start_stop=[4 * h//8, 6 * h//8],
                                   xy_window=(64, 64), xy_overlap=(0.8, 0.8))
    windows_multiscale.append(windows_64)

    windows_128 = slide_window(image, x_start_stop=[None, None], y_start_stop=[3 * h//8, h],
                                   xy_window=(128, 128), xy_overlap=(0.8, 0.8))
    windows_multiscale.append(windows_128)

    if verbose:
        windows_img_32 = draw_boxes(image, windows_32, color=(0, 0, 255), thick=1)
        windows_img_64 = draw_boxes(image, windows_64, color=(0, 255, 0), thick=1)
        windows_img_128 = draw_boxes(image, windows_128, color=(255, 0, 0), thick=1)

        stitching = stitch_together([windows_img_32, windows_img_64, windows_img_128], (1, 3), resize_dim=(1300, 500))
        cv2.imshow('', stitching)
        cv2.waitKey()

    return np.concatenate(windows_multiscale)


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     verbose=False, feature_vec=True):
    # Call with two outputs if vis==True
    if verbose:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=verbose, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=verbose, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()  # just re
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def image_to_features(image, feat_extr_params):

    color_space = feat_extr_params['color_space']
    spatial_size = feat_extr_params['spatial_size']
    hist_bins = feat_extr_params['hist_bins']
    orient = feat_extr_params['orient']
    pix_per_cell = feat_extr_params['pix_per_cell']
    cell_per_block = feat_extr_params['cell_per_block']
    hog_channel = feat_extr_params['hog_channel']
    spatial_feat = feat_extr_params['spatial_feat']
    hist_feat = feat_extr_params['hist_feat']
    hog_feat = feat_extr_params['hog_feat']

    image_features = []

    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(image)

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        image_features.append(spatial_features)

    if hist_feat:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        image_features.append(hist_features)

    if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     verbose=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, verbose=False, feature_vec=True)

        # Append the new feature vector to the features list
        image_features.append(hog_features)

    return np.concatenate(image_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_from_file_list(file_list, feat_extr_params):

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of image files
    for file in file_list:

        image = cv2.imread(file)

        # compute the features of this particular image, then append to the list
        file_features = image_to_features(image, feat_extr_params)
        features.append(file_features)

    return features


# Define a function that takes an image, start and stop positions in both x and y,
# window size (x and y dimensions), and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    n_x_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    n_y_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x / y
    n_x_windows = np.int(x_span / n_x_pix_per_step) - 1
    n_y_windows = np.int(y_span / n_y_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions.
    # Note: you could vectorize this step, but in practice you'll be considering windows one by one
    # with your classifier, so looping makes sense
    for i in range(n_y_windows):
        for j in range(n_x_windows):
            # Calculate window position
            start_x = j * n_x_pix_per_step + x_start_stop[0]
            end_x = start_x + xy_window[0]
            start_y = i * n_y_pix_per_step + y_start_stop[0]
            end_y = start_y + xy_window[1]

            # Append window position to list
            window_list.append(((start_x, start_y), (end_x, end_y)))

    # Return the list of windows
    return window_list


def draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6):
    """
    Draw all bounding boxes in `bbox_list` onto a given image.
    :param img: input image
    :param bbox_list: list of bounding boxes
    :param color: color used for drawing boxes
    :param thick: thickness of the box line
    :return: a new image with the bounding boxes drawn
    """
    # Make a copy of the image
    img_copy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bbox_list:
        # Draw a rectangle given bbox coordinates
        tl_corner = tuple(bbox[0])
        br_corner = tuple(bbox[1])
        cv2.rectangle(img_copy, tl_corner, br_corner, color, thick)

    # Return the image copy with boxes drawn
    return img_copy


# Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, feat_extraction_params):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        features = image_to_features(test_img, feat_extraction_params)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


if __name__ == '__main__':

    img = cv2.imread('test_images/test1.jpg')

    image_to_features(img)