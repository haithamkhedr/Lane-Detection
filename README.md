# Lane Detection for Self-Driving cars

## Camera Calibration

The code of the camera calibration is in `main/CameraCalibration.py`. The Calibration is done on Chess board test images by preparing object points and image points of the test images then use `cv2.calibrateCamera()` to calibrate the camera. The object points are the (x,y,z=0) coordiantes of the chess board corners in the world, and the image points are the 2D image coordinates of the corners in the image plane. After calculating the calibration parameters they are saved as a pickle file to `main/CalibrationParameters.p` to be loaded in the main notebook `main/Lane Detection.ipynb` in the 3rd cell. To undsitort images from the scene opencv `cv2.undistort()` is used and here are the results:
![Alt text] (./output_images/Calibration.jpg)

## Pipeline
The Pipeline is present in the 5th cell and helper functions in the 4th cell in `main/Lane Detection.ipynb`
I will explain my pipeline on this image:
![Alt text] (./test_images/test5.jpg)

#### 1-Image undistortion
After saving the Calibration parameters in the camera calibration phase, they can be used here to undistort images using `cv2.undistort()`
and here is the result:
![Alt text] (./output_images/Undistortion.jpg)

#### 2-Color transforms and gradients 

This is almost the most important part in the pipeline as it filters out unimportant colors as well as pixels which are most probably not lane lines. This is done by transforming the RGB image to HSL color space and to grayscale image. A gradient threshold(`thresh_gradient()` in 4th cell) is done on both new images and then they are merged to form the binary image. The last step is to mask this image to process only the region of interests(directly infront of the car where the lane would be). This part can be found in the 5th cell in`main/Lane Detection.ipynb`. The result is shown below:
![Alt text] (./output_images/Gradient_thresholding.jpg)
