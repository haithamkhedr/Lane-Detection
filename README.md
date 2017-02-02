# Lane Detection for Self-Driving cars

## Camera Calibration

The code of the camera calibration is in `main/CameraCalibration.py`. The Calibration is done on Chess board test images by preparing object points and image points of the test images then use `cv2.calibrateCamera()` to calibrate the camera. The object points are the (x,y,z=0) coordiantes of the chess board corners in the world, and the image points are the 2D image coordinates of the corners in the image plane. After calculating the calibration parameters they are saved as a pickle file to `main/CalibrationParameters.p` to be loaded in the main notebook `main/Lane Detection.ipynb` in the 3rd cell. To undsitort images from the scene opencv `cv2.undistort()` is used and here are the results:
![Alt text] (./output_images/Calibration.jpg)
