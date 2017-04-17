## Report

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report/car.jpg "car"
[image2]: ./report/notcar.jpg "notcar"
[image3]: ./report/hog_car.jpg "hog_car"
[image4]: ./report/hog_notcar.jpg "hog_notcar"
[image5]: ./report/slide_window.jpg "slide_window"
[image6]: ./report/find_cars.jpg "find_cars"
[image7]: ./report/heatmap.jpg "heatmap"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are 2 example(1 car and 1 not car) using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out that using the parameter below gives me the better result:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for feature extraction is contained in the 5th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".
I used convert_color, color_hist, HOG and bin_spatial function to extract feature vector from image


The code for trained a classifier is contained in the 13th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".
I trained a linear SVM using LinearSVC() from sklearn.svm package.
1. normalized data
2. shuffle data with np.random.randint()
3. split data into 80% training and 20% testing

Here is the training result:

Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
22.23 Seconds to train SVC...
Test Accuracy of SVC =  0.9918

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used slide_window() and search_windows() with overlapping parameter in the 7th and 8th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".

Here is a example for slide_window() without overlapping:

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I combined the single-image feature extraction, classifier prediction, slide_window() and search_windows() into 1 function called find_cars() in the 14th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".

Here are some example images:

![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 16th and 17th code cell of the IPython notebook located in "./Vehicle_Detection.ipynb".  

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


### Here is a frames and their corresponding heatmaps:

![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

From some heat map example and video result, although the accuracy is about 99%, I still can fount out some false positives. I think tuning some training parameter could solve this problem.

For the future work, I will combined the lane finding with vehicle detection these 2 project together.

