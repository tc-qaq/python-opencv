import cv2 as cv
import numpy as np

def edgeDetecte(src,dst,blurKsize=7,edgeKsize=5):
    if(blurKsize >= 3):
        blurredSrc = cv.medianBlur(src,blurKsize)
        graySrc    = cv.cvtColor(blurredSrc,cv.COLOR_BGR2GRAY)
    else:
        graySrc    = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    
    cv.Laplacian(graySrc, cv.CV_8U, graySrc, ksize= edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv.merge(channels, dst)
    print(channels)
    cv.imshow('ori',src)
    cv.imshow('edg',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
class VConvolutionFilter(object):    
    def __init__(self,kernel):
        self._kernel = kernel
    
    def apply(self,src,dst):
        cv.filter2D(src,-1,self._kernel,dst)
        
class SharpenFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-1,-1,-1],
                           [-1, 8,-1],
                           [-1,-1,-1]])
        VConvolutionFilter.__init__(self, kernel)
        
class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04]])
        VConvolutionFilter.__init__(self, kernel)

class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = np.array([[-2,-1,0],
                           [-1, 1,1],
                           [ 0, 1,2]])
        VConvolutionFilter.__init__(self, kernel)

