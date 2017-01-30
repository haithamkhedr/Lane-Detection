import numpy
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from CameraCalibration import *

#dict = pickle.load(open('CalibrationParameters.p','rb'))
calib_path = '../camera_cal/'
size = (720,1280)
dict = calibrate(calib_path, size , save = True,ignore_old_calib = True )
mtx = dict['mtx']
dist = dict ['dist']
files = glob.glob('../camera_cal/*.jpg')
undistorted_img =[]
img = []
for file in files:
    f,(ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))
    img = cv2.imread(file)
    undistorted_img = cv2.undistort(img,mtx,dist,None,mtx)
    ax1.imshow(img)
    ax2.imshow(undistorted_img)
    plt.show()
    input('')



