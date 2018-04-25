import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from distutils.command.check import check

def corners():
    img = cv.imread('../images/baboon.png')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray= np.float32(gray)
    dst = cv.cornerHarris(gray,2,23,0.04)
    
    img[dst>0.001 * dst.max()] = [0,0,25]
    cv.imshow('corners',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def scales_corner():
    img = cv.imread('../images/baboon.png')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift= cv.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray,None)
    
    img = cv.drawKeypoints(img,keypoints,img,(51,163,236),
                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('corners',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def fd(algorithm):
    img = cv.imread('../images/baboon.png')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    alg = None
    if(algorithm == "SIFT"):
        alg = cv.xfeatures2d.SIFT_create()
    elif(algorithm == "SURF"):
        alg = cv.xfeatures2d.SURF_create(4000)
    keypoints, descriptor = alg.detectAndCompute(gray,None)
    img = cv.drawKeypoints(img,keypoints,img,(51,163,236),
                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('corners',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# FAST (Features from Accelerated Segment Test)    
# draw a circle with 16 pixls at center (x,y) 
# consider the center and neigbor with thresh
# BRIEF (Binary Robust Independent Elementary Features)  descriptor
# -------------------------ORB------------------------------------
# add a component to FAST
# effective compute BRIEF
# variance and relation analytis based on BRIEF
# rotation_invari to learn unrelation BRIEF feature
def orb_implement():
    img1 = cv.imread('../images/tc1.png')
    img2 = cv.imread('../images/tc2.png')
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    kp1,des1 = orb.detectAndCompute(gray1,None)
    kp2,des2 = orb.detectAndCompute(gray2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING,True)
    matches = bf.match(des1, des2)
    matches = sorted(matches,key = lambda x:x.distance)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:40],img2,flags=2)
    plt.imshow(img3)
    plt.show()

def KNN_match():
    img1 = cv.imread('../images/tc1.png')
    img2 = cv.imread('../images/tc2.png')
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
    orb = cv.ORB_create()
    kp1,des1 = orb.detectAndCompute(gray1,None)
    kp2,des2 = orb.detectAndCompute(gray2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING,True)
    
    matches = bf.knnMatch(des1, des2,k=1)    
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches[:10],img2,flags=2)
    img3 = cv.cvtColor(img3,cv.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()
# FLANN (fast library for approximate NN)
def FLANN_imp():
    img1 = cv.imread('../images/tc1.png')
    img2 = cv.imread('../images/tc2.png')
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(gray1,None)
    kp2,des2 = sift.detectAndCompute(gray2,None)
    
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=15)  
    searchParams = dict(checks=150)  
    
    flann = cv.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(des1,des2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    
    for i,(m,n) in enumerate(matches):
        if (m.distance < 0.5*n.distance):
            matchesMask[i] = [1,0]
    drawParams = dict(matchColor = (0,255,0),
                      singlePointColor = (255,0,0),
                      matchesMask = matchesMask,
                      flags = 0)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**drawParams)
    plt.imshow(img3)
    plt.show()
   
# 单应性 hamography： 对比两幅图像，其中一副出现投影扭曲时,仍可以匹配 
def FLANN_hamography():
    img1 = cv.imread('../images/tc1.png')
    img2 = cv.imread('../images/tc2.png')
    gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
    sift = cv.xfeatures2d.SIFT_create()
    kp1,des1 = sift.detectAndCompute(gray1,None)
    kp2,des2 = sift.detectAndCompute(gray2,None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  
    searchParams = dict(checks=50)      
    flann = cv.FlannBasedMatcher(indexParams,searchParams)
    matches = flann.knnMatch(des1,des2, k=2)
    
    matchesMask = [[0,0] for i in range(len(matches))]
    
    for i,(m,n) in enumerate(matches):
        if (m.distance < 0.5*n.distance):
            matchesMask[i] = [1,0]
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3)
    plt.show()
    
if __name__ == '__main__':
#     corners()
#     scales_corner()
#     fd("SIFT")
#     fd("SURF")
#     orb_implement()
#     KNN_match()
#     FLANN_imp()
    FLANN_hamography()