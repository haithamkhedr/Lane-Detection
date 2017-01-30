#############Module for Camera calibartion##############

# Imports
import glob
import cv2
import numpy as np
import pickle

def calibrateCamera(objpts , imgpts , size):

    ret,mtx,dist,_,_ =  cv2.calibrateCamera(objpts , imgpts, size,None,None)
    if (ret):
       dict = {}
       dict['mtx'] = mtx
       dict['dist'] = dist
       return dict



def calibrate(path ,size, save = True,ignore_old_calib = True):
    if(ignore_old_calib == False):
        lst = glob.glob('./*.p')
        if(lst):
            return  pickle.load(open('CalibrationParameters.p','rb'))
    objpts = []
    imgpts = []
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
    files = glob.glob(path + '/*.jpg')
    for imgFile in files:
        img = cv2.imread(imgFile)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret , corners = cv2.findChessboardCorners(gray, (8,6),None)
        if(ret == True):
            objpts.append(objp)
            imgpts.append(corners)
    dict = calibrateCamera(objpts , imgpts , size)
    if (save == True):
        pickle.dump(dict , open('CalibrationParameters.p','wb'))
    return dict

#calibrate('../camera_cal',(540,960))

