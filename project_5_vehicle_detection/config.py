# root directory that contain all vehicle images in nested subdirectories
root_data_vehicle = '../../../NANODEGREE/term_1/project_5_vehicle_detection/vehicles'

# root directory that contain all NON-vehicle images in nested subdirectories
root_data_non_vehicle = '../../../NANODEGREE/term_1/project_5_vehicle_detection/non-vehicles'

feat_extraction_params = {'color_space': 'RGB',       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                          'orient': 9,                # HOG orientations
                          'pix_per_cell': 8,          # HOG pixels per cell
                          'cell_per_block': 2,        # HOG cells per block
                          'hog_channel': "ALL",       # Can be 0, 1, 2, or "ALL"
                          'spatial_size': (16, 16),   # Spatial binning dimensions
                          'hist_bins': 16,            # Number of histogram bins
                          'spatial_feat': True,       # Spatial features on or off
                          'hist_feat': True,          # Histogram features on or off
                          'hog_feat': True}           # HOG features on or off


