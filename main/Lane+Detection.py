
# coding: utf-8

# ## Lane detection

# ## First :  Camera Calibration 

# In[1]:

#Imports
import numpy
import cv2
import glob
import matplotlib.pyplot as plt
from CameraCalibration import *
from math import pi


# In[2]:

calib_path = '../camera_cal/'
test_path = '../test_images'
size = (720,1280)
RThresh = (200,255)
SThresh = (80,255)
GThresh = (80,255)


# In[3]:

dict = calibrate(calib_path, size , save = True,ignore_old_calib = True )
mtx = dict['mtx']
dist = dict ['dist']
print('Loaded Calibration Matrix')


# In[4]:

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    #masked_image = cv2.bitwise_and(img, mask)
    #masked_image [masked_image > 0] = 1
    return mask

def thresh_gradient(img, kernel=3 , threshold=(0,255),angle = pi/2):
    
    sobelx = cv2.Sobel(img ,cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img ,cv2.CV_64F, 0, 1, ksize=kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    theta = np.arctan2(np.abs(sobely),np.abs(sobelx))
    scale = 255.0 / np.max(mag)
    mag = (mag * scale).astype(np.uint8)
    binary = np.zeros_like(mag)
    binary[(mag > threshold[0]) & (mag <= threshold[1]) & (theta < angle)] = 1
    #binary = cv2.Canny(img,threshold[0],threshold[1])
    return binary  
def thresh_color(img,threshold = (0,255)):
    
    binary = np.zeros_like(img)
    binary [(img > threshold[0]) & (img <= threshold[1])] = 1
    binary = binary.astype(np.uint8)
    return binary

def find_lane_pixels(img):
    imshape = img.shape
    dict = {}
    leftX = []
    leftY = []
    rightX = []
    rightY = []
    x_left = 0
    x_right =0
    histogram1 = np.sum(img[int(img.shape[0]/1.1):,0:int(img.shape[1]/1.6)], axis = 0)
    histogram2 = np.sum(img[int(img.shape[0]/1.1):,int(img.shape[1]/1.6):], axis = 0)
    x_left = np.argmax(histogram1)
    x_right = np.argmax(histogram2) + int(img.shape[1]/1.7)
    #print(x_left,x_right)
#     idx = np.argsort(histogram)
#     if(idx[-1] < idx[-2]):
#         x_left = idx[-1]
#         x_right = idx[-2]
#     else:
#         x_left  =idx[-2]
#         x_right =idx[-1]
        
    L =15
    W =40
    for yi in range(imshape[0]-1,L,-L):
        #print(yi-L,int(x_left-W/2),'-->',yi,int(x_left+W/2))
        window = img[yi-L:yi,int(x_left-W/2):int(x_left+W/2)]
        y,x = np.where(window ==1 )
        x +=  int(x_left-W/2)
        y+= yi-L
        if(x.size is not 0):
            leftX.append(x.tolist())
            leftY.append(y.tolist())
            x_left = int(np.mean(x))
        
        window = img[yi-L:yi,int(x_right-W/2):int(x_right+W/2)]
        y,x = np.where(window ==1 )
        x +=  int(x_right-W/2)
        y += yi-L
        if(x.size is not 0):
            rightX.append(x.tolist())
            rightY.append(y.tolist())
            x_right = int(np.mean(x))
        
    dict['leftX'] = leftX
    dict['rightX']= rightX
    dict['leftY'] = leftY
    dict['rightY'] = rightY
    
    return dict


        
    
def fit_lanes(dict,imshape):
        
    yvals = np.arange(imshape[0])
    leftX = dict['leftX']
    leftY = dict['leftY']
    rightX = dict['rightX']
    rightY = dict['rightY']
    leftX = [i for list in leftX for i in list]
    leftY = [i for list in leftY for i in list] 
    rightX = [i for list in rightX for i in list]
    rightY = [i for list in rightY for i in list]
    Lanes = np.zeros(imshape).astype(np.uint8)
    #Lanes = np.dstack((Lanes,Lanes,Lanes))
    leftfit = np.polyfit(leftY,leftX,2)
    leftfitX = leftfit[0] * yvals**2 + leftfit[1] * yvals + leftfit[2]
    rightfit = np.polyfit(rightY,rightX,2)
    rightfitX = rightfit[0] * yvals+ rightfit[1] * yvals + rightfit[2]
    rightfitX[rightfitX >= imshape[1]] = 0
    rightfitX[rightfitX < 0] = 0
    pts_left = np.array([np.transpose(np.vstack([leftfitX, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightfitX, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    Lanes[np.arange(imshape[0]).tolist(),rightfitX.tolist()] =255
    Lanes[np.arange(imshape[0]).tolist(),leftfitX.tolist()] = 255
    #Lanes = cv2.fillPoly(Lanes, np.int_([pts]), (0,255, 0))
    
    return Lanes
    
    
def perspective_transform(img):
    shape = img.shape
    p1 = [int(0.2*shape[1]),shape[0]]
    p2 = [int(0.48*shape[1]),int(0.62*shape[0])]
    p3 = [int(0.6*shape[1]),int(0.62*shape[0])]
    p4 = [int(0.9*shape[1]),shape[0]]
    src = np.float32([p1,p2,p3,p4])
    p1 = [int(0.25*shape[1]),shape[0]]
    p2 = [int(0.3*shape[1]),0]
    p3 = [int(0.93*shape[1]),0]
    p4 = [int(0.68*shape[1]),shape[0]]
    dst = np.float32([p1,p2,p3,p4])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped,M,Minv


# In[6]:

test_files = glob.glob('../test_images/*.jpg')

for file in test_files:
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (20,10))
    imgRGB = cv2.imread(file)
    imgRGB = cv2.undistort(imgRGB,mtx,dist)
    imshape = imgRGB.shape
    p1 = (0,imshape[0])
    p2 = (int(0.5*imshape[1]),int(0.5*imshape[0]))
    p3 = (int(0.51*imshape[1]),int(0.5*imshape[0]))
    p4 = (imshape[1],imshape[0])
    vertices = np.array([[p1,p2, p3,p4]], dtype=np.int32) 
    imgHLS = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2HLS)
    imgGray = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY)
    masked_img = region_of_interest(imgGray,vertices)
    #RChannel = imgRGB[:,:,2]
    SChannel = imgHLS[:,:,2]
  
    sBinary = thresh_color(SChannel,SThresh)
    binary_gradient = thresh_gradient(imgGray,kernel = 3 ,threshold = GThresh)
    merged_binary = np.zeros_like(binary_gradient)
    merged_binary [(binary_gradient ==1) | (sBinary ==1)] = 1
    #merged_binary &= masked_img[:,:,0]
    merged_binary  = np.bitwise_and(merged_binary,masked_img)
    Sobely = cv2.Sobel(sBinary,cv2.CV_64F,0,1,3)
    Sobely = np.abs(Sobely)
    scale = 255.0/np.max(Sobely)
    Sobely = (Sobely*scale).astype(np.uint8)
    merged_binary[Sobely > 100] =0
    perTransform,M,Minv = perspective_transform(merged_binary)
    dict = find_lane_pixels(perTransform)
    rightX = dict['rightX']
    rightY = dict['rightY']
    leftY = dict['leftY']
    leftX = dict['leftX']
    leftX = [i for list in leftX for i in list]
    leftY = [i for list in leftY for i in list] 
    rightX = [i for list in rightX for i in list]
    rightY = [i for list in rightY for i in list]
    
#     leftX = dict['leftX']
#     leftY = dict['leftY']
#     rightX = dict['rightX']
#     rightY = dict['rightY']
#     leftX = [i for list in leftX for i in list]
#     leftY = [i for list in leftY for i in list] 
#     rightX = [i for list in rightX for i in list]
#     rightY = [i for list in rightY for i in list]
    Lanes = fit_lanes(dict,imshape)
    #Lanes = cv2.warpPerspective(Lanes,Minv,(imshape[1],imshape[0]))
    image = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2RGB)
    final_out = cv2.addWeighted(image, 1, Lanes, 0.3, 0)

    

    
   
#     ax1.imshow(imgRGB[:,:,2],cmap = 'gray')
#     ax2.imshow(imgHLS[:,:,2],cmap = 'gray')
    ax1.imshow(perTransform,cmap = 'gray')
    ax2.imshow(Lanes,cmap = 'gray')
    ax3.imshow(final_out)
    ax1.set_title('Perspective Transform')
    ax2.set_title('S channel Threshold')
    plt.show()


# In[ ]:

def process_image(img):
    f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = (20,10))
    #imgRGB = cv2.imread(file)
    #imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgRGB=img
    imgRGB = cv2.undistort(imgRGB,mtx,dist)
    imshape = imgRGB.shape
    p1 = (0,imshape[0])
    p2 = (int(0.5*imshape[1]),int(0.5*imshape[0]))
    p3 = (int(0.51*imshape[1]),int(0.5*imshape[0]))
    p4 = (imshape[1],imshape[0])
    vertices = np.array([[p1,p2, p3,p4]], dtype=np.int32) 
    imgHLS = cv2.cvtColor(imgRGB,cv2.COLOR_RGB2HLS)
    imgGray = cv2.cvtColor(imgRGB,cv2.COLOR_RGB2GRAY)
    masked_img = region_of_interest(imgGray,vertices)
    #RChannel = imgRGB[:,:,2]
    SChannel = imgHLS[:,:,2]
  
    sBinary = thresh_color(SChannel,SThresh)
    binary_gradient = thresh_gradient(imgGray,kernel = 3 ,threshold = GThresh)
    merged_binary = np.zeros_like(binary_gradient)
    merged_binary [(binary_gradient ==1) | (sBinary ==1)] = 1
    #merged_binary &= masked_img[:,:,0]
    merged_binary  = np.bitwise_and(merged_binary,masked_img)
    Sobely = cv2.Sobel(sBinary,cv2.CV_64F,0,1,3)
    Sobely = np.abs(Sobely)
    scale = 255.0/np.max(Sobely)
    Sobely = (Sobely*scale).astype(np.uint8)
    merged_binary[Sobely > 100] =0
    perTransform,M,Minv = perspective_transform(merged_binary)
    dict = find_lane_pixels(perTransform)
#     leftX = dict['leftX']
#     leftY = dict['leftY']
#     rightX = dict['rightX']
#     rightY = dict['rightY']
#     leftX = [i for list in leftX for i in list]
#     leftY = [i for list in leftY for i in list] 
#     rightX = [i for list in rightX for i in list]
#     rightY = [i for list in rightY for i in list]
    Lanes = fit_lanes(dict,imshape)
    Lanes = cv2.warpPerspective(Lanes,Minv,(imshape[1],imshape[0]))
    final_out = cv2.addWeighted(imgRGB, 1, Lanes, 0.3, 0)
    return final_out


# In[ ]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
out_video = 'out_challenge.mp4'
clip1 = VideoFileClip("../challenge_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(out_video, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(out_video))


# In[ ]:



