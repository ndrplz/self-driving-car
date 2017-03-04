
# root directory that contain all vehicle images in nested subdirectories
root_data_vehicle = '../../../NANODEGREE/term_1/project_5_vehicle_detection/vehicles'

# root directory that contain all NON-vehicle images in nested subdirectories
root_data_non_vehicle = '../../../NANODEGREE/term_1/project_5_vehicle_detection/non-vehicles'

# parameters used in the phase of feature extraction
feat_extraction_params = {'resize_h': 64,             # resize image height before feat extraction
                          'resize_w': 64,             # resize image height before feat extraction
                          'color_space': 'YCrCb',     # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                          'orient': 9,                # HOG orientations
                          'pix_per_cell': 8,          # HOG pixels per cell
                          'cell_per_block': 2,        # HOG cells per block
                          'hog_channel': "ALL",       # Can be 0, 1, 2, or "ALL"
                          'spatial_size': (32, 32),   # Spatial binning dimensions
                          'hist_bins': 16,            # Number of histogram bins
                          'spatial_feat': True,       # Spatial features on or off
                          'hist_feat': True,          # Histogram features on or off
                          'hog_feat': True}           # HOG features on or off



# color_space = 'YcrCb'
# spatial_size = (32, 32)
# add 96, 96 windows size
# ystartm ystop = 400, 656

# compute the hog feature on the whole area of interest, then subsample

# add a scale parameter instead of sampling different windows size (around min 49)
#     indeed, if you keep the windows size the same but rescale the image, is like having different windows sizes




