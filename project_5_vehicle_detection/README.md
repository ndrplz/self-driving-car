# Vehicle Detection Project

<p align="center">
 <a href="https://www.youtube.com/watch?v=Cd7p5pnP3e0"><img src="./img/overview.gif" alt="Overview" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p>

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Abstract

The goal of the project was to develop a pipeline to reliably detect cars given a video from a roof-mounted camera: in this readme the reader will find a short summary of how I tackled the problem.

**Long story short**:
 - (baseline) HOG features + linear SVM to detect cars, temporal smoothing to discard false positive
 - (submission) [SSD deep network](https://arxiv.org/pdf/1512.02325.pdf) for detection, thresholds on detection confidence and label to discard false positive 
 
*That said, let's go into details!*

### Good old CV: Histogram of Oriented Gradients (HOG)

#### 1. Feature Extraction.

In the field of computer vision, a *features* is a compact representation that encodes information that is relevant for a given task. In our case, features must be informative enough to distinguish between *car* and *non-car* image patches as accurately as possible.

Here is an example of how the `vehicle` and `non-vehicle` classes look like in this dataset:

<p align="center">
  <img src="./img/noncar_samples.png" alt="non_car_img">
  <br>Randomly-samples non-car patches.
</p>

<p align="center">
  <img src="./img/car_samples.png" alt="car_img">
  <br>Randomly-samples car patches.
</p>

The most of the code that relates to feature extraction is contained in [`functions_feat_extraction.py`](functions_feat_extraction.py). Nonetheless, all parameters used in the phase of feature extraction are stored as dictionary in [`config.py`](config.py), in order to be able to access them from anywhere in the project.

Actual feature extraction is performed by the function `image_to_features`, which takes as input an image and the dictionary of parameters, and returns the features computed for that image. In order to perform batch feature extraction on the whole dataset (for training), `extract_features_from_file_list` takes as input a list of images and return a list of feature vectors, one for each input image.

For the task of car detection I used *color histograms* and *spatial features* to encode the object visual appearence and HOG features to encode the object's *shape*. While color the first two features are easy to understand and implement, HOG features can be a little bit trickier to master.

#### 2. Choosing HOG parameters.

HOG stands for *Histogram of Oriented Gradients* and refer to a powerful descriptor that has met with a wide success in the computer vision community, since its [introduction](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) in 2005 with the main purpose of people detection. 

<p align="center">
  <img src="./img/hog_car_vs_noncar.jpg" alt="hog" height="128">
  <br>Representation of HOG descriptors for a car patch (left) and a non-car patch (right).
</p>

The bad news is, HOG come along with a *lot* of parameters to tune in order to work properly. The main parameters are the size of the cell in which the gradients are accumulated, as well as the number of orientations used to discretize the histogram of gradients. Furthermore, one must specify the number of cells that compose a block, on which later a feature normalization will be performed. Finally, being the HOG computed on a single-channel image, arises the need of deciding which channel to use, eventually computing the feature on all channels then concatenating the result.

In order to select the right parameters, both the classifier accuracy and computational efficiency are to consider. After various attemps, I came up to the following parameters that are stored in [`config.py`](config.py):
```
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
```

#### 3. Training the classifier

I trained a linear SVM using...

### Computer Vision on Steroids, a.k.a. Deep Learning

#### 1. SSD (*Single Shot Multi-Box Detector*) network

### Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:



### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:


### Here the resulting bounding boxes are drawn onto the last frame in the series:




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

