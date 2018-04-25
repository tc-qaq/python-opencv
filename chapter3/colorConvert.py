import cv2 as cv
import math
import numpy as np
import scipy.ndimage
from scipy import ndimage
import matplotlib.pyplot as plt 

def rgb2hsi():
    img = cv.imread('../images/baboon.png')
    b   = (img[:,:,0]/255.0) 
    g   = (img[:,:,1]/255.0)
    r   = (img[:,:,2]/255.0)
    
    num = 0.5*((r-g)+(r-b))
    den = np.sqrt((r-g)**2+(r-b)*(g-b))
    theta=np.arccos(num/(den+0.0000000000000000001)) * 180 / 3.14
    size = img.shape
    
    condition = np.array(b<=g)
    H = np.where(condition,(theta / 360 * 255).astype(int),((360 - theta) / 360 * 255).astype(int))
    num = np.minimum(np.minimum(r,g),b)
    den = r+g+b+0.0000000000000000001
    S   = (1 - (num * 3) / den) * 255
    I   = (den/3) * 255
    img1 = np.ones((512,512,3),int);
    img1[:,:,0] = H
    img1[:,:,1] = S
    img1[:,:,2] = I

    img1 = np.array(img1).reshape(512,512,3)
    print(img1.shape)
    print(img[100][100])
    print(img1[100][100])
    cv.imshow("ImageRGB",img)
    cv.imshow("ImageHSI",img1)    
    cv.waitKey(0)
    cv.destroyAllWindows() 
    
def high_pass():
    kernel_3x3 = np.array([[-1,-1,-1],
                           [-1, 8,-1],
                           [-1,-1,-1]])
    kernel_5x5 = np.array([[-1,-1,-1,-1,-1],
                           [-1, 1, 2, 1,-1],
                           [-1, 2, 4, 2,-1],
                           [-1, 1, 2, 1,-1],
                           [-1,-1,-1,-1,-1]])
    
    img = cv.imread('../images/lena.png',0)
    k3  = ndimage.convolve(img,kernel_3x3)
    k5  = ndimage.convolve(img,kernel_5x5)
    
    blurred = cv.GaussianBlur(img,(11,11),0)
    g_hpf   = img - blurred
    
    cv.imshow('ori',img)
    cv.imshow('3x3',k3)
    cv.imshow('5x5',k5)
    cv.imshow('g_hpf',g_hpf)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def defineK():
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    src = cv.imread('../images/lena.png')
    dst = cv.filter2D(src,-1,kernel)
    cv.imshow('ori',src)
    cv.imshow('dst',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def cannyEdge():
    img = cv.imread('../images/lena.png',0)
    img1 = cv.Canny(img,img.shape[0],img.shape[1])
    
    cv.imshow('ori',img)
    cv.imshow('canny',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def contour():
    img = cv.imread('../images/lena.png',0)
    cv.imshow('ori',img)
    ret,thresh = cv.threshold(img,127,255,0)
    img1,contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    color = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    img   = cv.drawContours(color,contours,-1,(0,255,0),2)
    cv.imshow('contours',color)
    
    cv.imshow('img1',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rect_circle_contours():
    # downsampling pyrDown
    # upsampling   pyrUp 
    #img = cv.pyrDown(cv.imread('../images/airplane.png'))
    #img1 = cv.pyrUp(cv.imread('../images/airplane.png'),cv.IMREAD_UNCHANGED)    
    img = cv.imread('../images/peppers.png')
    color = cv.cvtColor(img.copy(),cv.COLOR_BGR2GRAY)
    rect,thresh = cv.threshold(color,128,255,cv.THRESH_BINARY)
    image,contours,hier = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)    
    for c in contours:
        #get bound
        x,y,w,h = cv.boundingRect(c)
        #draw rect
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
        #min bound
        rect = cv.minAreaRect(c)
        box  = cv.boxPoints(rect)
        box  = np.int0(box)
        cv.drawContours(img,[box],0,(0,0,255),5)
        #get circle
        (x,y),radius = cv.minEnclosingCircle(c)
        center = int(x),int(y)
        radius = int(radius)
        #draw circle
        cv.circle(img,center,radius,(255,0,0),5)
    cv.drawContours(img,contours,-1,(255,0,0),1)
    cv.imshow('downsampling',img)
    #cv.imshow('upsampling',img1)
    
    cv.waitKey()
    cv.destroyAllWindows()
    
def convexContour():
    img = cv.imread('../images/peppers.png')
    color = cv.cvtColor(img.copy(),cv.COLOR_BGR2GRAY)
    rect,thresh = cv.threshold(color,128,255,cv.THRESH_BINARY)
    image,contours,hier = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)    
    for c in contours:
        print('c',c)
        epsilon = 0.01 * cv.arcLength(c,True);
        approx  = cv.approxPolyDP(c,epsilon,True)
        print('approx',approx)
        cv.polylines(img,[approx],True,(255,0,0),2)
        hull    = cv.convexHull(c)
        print('hull',hull)
        cv.polylines(img,[hull],True,(0,255,0),3)
    #cv.drawContours(img,contours,-1,(255,0,0),1)
    cv.imshow('downsampling',img)   
    
    cv.waitKey()
    cv.destroyAllWindows()

def lineDetect():
    img  = cv.imread('../images/geo.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges= cv.Canny(gray,50,120)
    minLineLength = 10
    maxLineGap    = 5
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    print(lines[0])
    for line in lines:
        x1,y1,x2,y2 = line[0][0],line[0][1],line[0][2],line[0][3]
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),5)
    cv.imshow('edges',edges)
    cv.imshow('lines',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def circleDetect():
    img  = cv.imread('../images/geo.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray,5)
    
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,120,
                              param1=100,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0,:]:
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        cv.circle(img,(i[0],i[1]),2,(0,0,255),3)    
    cv.imshow('ori',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def frontDetect():
    img  = cv.imread('../images/lena.png')
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,512,512)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,20,cv.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0),0,1).astype('uint8')
    img   = img * mask2[:,:,np.newaxis]
    
    plt.subplot(121), plt.imshow(img)
    plt.title('grabCut'),plt.xticks([]),plt.yticks([])
    plt.subplot(122), plt.imshow(cv.cvtColor(
        cv.imread('../images/lena.png'),cv.COLOR_BGR2RGB))
    plt.title('original'),plt.xticks([]),plt.yticks([])
    plt.show()
    
def segment():
    img  = cv.imread('../images/airplane.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening= cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,2)
    sure_bg= cv.dilate(opening,kernel,3)
    
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg   = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    sure_fg = np.uint8(sure_fg) 
    unknown = cv.subtract(sure_bg,sure_fg)
    
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    
    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    plt.imshow(img)
    plt.show()
    
    
if __name__ == "__main__":
#     rgb2hsi()
#     high_pass()
#     defineK()
#     cannyEdge()
#     contour()
#     rect_circle_contours()
#     convexContour()
#     lineDetect()
#     circleDetect()
#     frontDetect()
    segment()