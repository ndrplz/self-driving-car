# Project 4 - Advanced Lane Finding

<p align="center">
 <a href="https://www.youtube.com/watch?v=g5BhDtoheE4"><img src="./img/overview.gif" alt="Overview" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p>

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_overview]: ./img/overview.gif "Output Overview"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view)

### Camera Calibration

OpenCV provide some really helpful built-in functions for the task on camera calibration. First of all, to detect the calibration pattern in the [calibration images](./camera_cal/), we can use the function `cv2.findChessboardCorners(image, pattern_size)`. 

Once we have stored the correspondeces between 3D world and 2D image points for a bunch of images, we can proceed to actually calibrate the camera through `cv2.calibrateCamera()`. Among other things, this function returns both the *camera matrix* and the *distortion coefficients*, which we can use to undistort the frames.

The code for this steps can be found in [calibration_utils](calibration_utils.py).   

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result (appreciating the effect of calibration is easier on the borders of the image): 

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./img/calibration_before.jpg" alt="calibration_before" width="60%" height="60%">
           <br>Chessboard image before calibration
      </p>
    </th>
    <th>
      <p align="center">
           <img src="./img/calibration_after.jpg" alt="calibration_after" width="60%" height="60%">
           <br>Chessboard image after calibration
      </p>
    </th>
  </tr>
</table>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera is calibrated, we can use the camera matrix and distortion coefficients we found to undistort also the test images. Indeed, if we want to study the *geometry* of the road, we have to be sure that the images we're processing do not present distortions. Here's the result of distortion-correction on one of the test images:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./img/test_calibration_before.jpg" alt="calibration_before" width="60%" height="60%">
           <br>Test image before calibration
      </p>
    </th>
    <th>
      <p align="center">
           <img src="./img/test_calibration_after.jpg" alt="calibration_after" width="60%" height="60%">
           <br>Test image after calibration
      </p>
    </th>
  </tr>
</table>

In this case appreciating the result is slightly harder, but we can notice nonetheless some difference on both the very left and very right side of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Correctly creating the binary image from the input frame is the very first step of the whole pipeline that will lead us to detect the lane. For this reason, I found that is also one of the most important. If the binary image is bad, it's very difficult to recover and to obtain good results in the successive steps of the pipeline. The code related to this part can be found [here](./binarization_utils.py).

I used a combination of color and gradient thresholds to generate a binary image. In order to detect the white lines, I found that [equalizing the histogram](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) of the input frame before thresholding works really well to highlight the actual lane lines. For the yellow lines, I employed a threshold on V channel in [HSV](http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html) color space. Furthermore, I also convolve the input frame with Sobel kernel to get an estimate of the gradients of the lines. Finally, I make use of [morphological closure](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html) to *fill the gaps* in my binary image. Here I show every substep and the final output:
<p align="center">
  <img src="./img/binarization.png" alt="binarization overview" width="90%" height="90%">
</p>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code relating to warping between the two perspective can be found [here](./perspective_utils.py). The function `calibration_utils.birdeye()` takes as input the frame (either color or binary) and returns the bird's-eye view of the scene. In order to perform the perspective warping, we need to map 4 points in the original space and 4 points in the warped space. For this purpose, both source and destination points are *hardcoded* (ok, I said it) as follows:

```
    h, w = img.shape[:2]

    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [546, 460],   # tl
                      [732, 460]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<p align="center">
  <img src="./img/perspective_output.png" alt="birdeye_view" width="90%" height="90%">
</p>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify which pixels of a given binary image belong to lane-lines, we have (at least) two possibilities. If we have a brand new frame, and we never identified where the lane-lines are, we must perform an exhaustive search on the frame. This search is implemented in `line_utils.get_fits_by_sliding_windows()`: starting from the bottom of the image, precisely from the peaks location of the histogram of the binary image, we slide two windows towards the upper side of the image, deciding which pixels belong to which lane-line.

On the other hand, if we're processing a video and we confidently identified lane-lines on the previous frame, we can limit our search in the neiborhood of the lane-lines we detected before: after all we're going at 30fps, so the lines won't be so far, right? This second approach is implemented in `line_utils.get_fits_by_previous_fits()`. In order to keep track of detected lines across successive frames, I employ a class defined in `line_utils.Line`, which helps in keeping the code cleaner.

```
class Line:

    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None
    
    ... methods ...
```

The actual processing pipeline is implemented in function `process_pipeline()` in [`main.py`](./main.py). As it can be seen, when a detection of lane-lines is available for a previous frame, new lane-lines are searched through `line_utils.get_fits_by_previous_fits()`: otherwise, the more expensive sliding windows search is performed.

The qualitative result of this phase is shown here:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./img/sliding_windows_before.png" alt="sliding_windows_before" width="60%" height="60%">
           <br>Bird's-eye view (binary)
      </p>
    </th>
    <th>
      <p align="center">
           <img src="./img/sliding_windows_after.png" alt="sliding_windows_after" width="60%" height="60%">
           <br>Bird's-eye view (lane detected)
      </p>
    </th>
  </tr>
</table>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Offset from center of the lane is computed in `compute_offset_from_center()` as one of the step of the procecssing pipeline defined in [`main.py`](./main.py). The offset from the lane center can be computed under the hypothesis that the camera is fixed and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation from the lane center as the distance between the center of the image and the midpoint at the bottom of the image of the two lane-lines detected.  

During the previous lane-line detection phase, a 2nd order polynomial is fitted to each lane-line using `np.polyfit()`. This function returns the 3 coefficients that describe the curve, namely the coefficients of both the 2nd and 1st order terms plus the bias. From this coefficients, following [this](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) equation, we can compute the radius of curvature of the curve. From an implementation standpoint, I decided to move this methods as properties of `Line` class.

```
class Line:
  ... other stuff before ...
    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The whole processing pipeline, which starts from input frame and comprises undistortion, binarization, lane detection and de-warping back onto the original image, is implemented in function `process_pipeline()` in [`main.py`](./main.py).

The qualitative result for one of the given test images follows:

<p align="center">
     <img src="./output_images/test2.jpg" alt="output_example" width="60%" height="60%">
     <br>Qualitative result for test2.jpg
</p>

All other test images can be found in [./output_images/](./output_images/)

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=g5BhDtoheE4).

---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I find that the more delicate aspect of the pipeline is the first step, namely the binarization of the input frame. Indeed, if that step fails, most of successive steps will lead to poor results. The bad news is that this part is implemented by thresholding the input frame, so we let the correct value of a threshold be our single-point of failure. This is *bad*! Being currently 2017, I think a CNN could be employed to successfully make this step more robust. Some datasets like [Synthia](http://synthia-dataset.net/) should hopefully  provide enough lane marking annotation to train a deep network. I must try this later :-)


