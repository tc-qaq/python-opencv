import cv2   as cv
import numpy as np
import time

# 
class CaptureManager(object):
    def __init__(self,capture,previewWindowManager=None,shouldMirrorPreview = False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview  = shouldMirrorPreview
        
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoWriter   = None
        
        self._startTime     = None
        self._framesElapsed = int(0)
        self._fpsEstimate   = None
        
    @property
    def channel(self):    
        return self._channel
    @channel.setter
    def channel(self,value):
        if(self._channel != value):
            self._channel = value
            self._frame = None
    
    @property
    def frame(self):
        if(self._enteredFrame and self._frame is None):
            _, self._frame = self._capture.retrieve()
            return self._frame
    
    @property
    def isWrittingImage(self):
        return self._imageFilename is not None
    
    @property
    def isWrittingVideo(self):
        return self._videoFilename is not None
    
    def enterFrame(self):
        # capture the next frame, and check that any previous frame was exited
        assert not self._enteredFrame, 'previous enterFrame() had not matching exitFrame()'
        if (self._capture is not None):
            self._enteredFrame = self._capture.grab()
    
    def exitFrame(self):
        # draw to the window. write to files release the frame
        # check whether any grabbed frame is retrieveable
        if (self.frame is None):
            self._enteredFrame = False
            return
        # update the FPS　estimate and related variables
        if (self._framesElapsed == 0):
            self._startTime = time.time()
        else:
            timeElapsed =time.time() - self._startTime
        self._fpsEstimate = self._framesElapsed/timeElapsed
    
        self._framesElapsed += 1
        # draw to window if any
        if (self.previewWindowManager is not None):
            if (self.shouldMirrorPreview):
                mirroredFram = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFram)
            else:
                self.previewWindowManager.show(self._frame)
        # write to the image file, if any
        if (self.isWrittingImage):
            cv.imwrite(self._imageFilename,self._frame)
            self._imageFilename = None
        # write to the video file, if any
        if (self.isWrittingVideo):
            self._writeVideoFrame()
            self._frame = None
            self._enteredFrame = False
    
    def writeImage(self,filename):
        self._imageFilename = filename
    
    def startWrittingVideo(self,filename,encoding=cv.VideoWriter_fourcc('I','4','2','0')):
        self._videoFilename = filename
        self._videoEncoding = encoding
    
    def stopWrittingVideo(self):
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter   = None
        
    def _writeVideoFrame(self):
        if (not self._isWrittingVideo):
            return
        if (self._videoWriter is None):
            fps = self._capture.get(cv.CAP_PROP_FPS)
            if (fps == 0.0):
                if (self._framesElapsed < 20):
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv.VideoWriter(
                                   self._videoFilename,self._videoEncoding,fps,size)
            self._videoWriter.write(self._frame)

 
class WindowManager(object):
    def __init__(self, windowName,keypressCallback = None):
        self._keypressCallback = keypressCallback
        
        self._windowName = windowName
        self._isWindowCreated = False
        
    def isWindowCreated(self):
        return self._isWindowCreated
    
    def createWindow(self):
        cv.namedWindow(self._windowName)
        self._isWindowCreated = True
        
    def show(self,frame):
        cv.imshow(self._windowName, frame)
        
    def destroyWindow(self):    
        cv.destroyWindow(self._windowName)
        self._isWindowCreated = False
        
    def processEvent(self):    
        keycode = cv.waitKey(1)
        if (self._keypressCallback is not None and keycode != -1):
            keycode &= 0xFF
            self._keypressCallback(keycode)
    
    