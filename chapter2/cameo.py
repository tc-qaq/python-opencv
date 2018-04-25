import cv2 as cv
from managers import WindowManager,CaptureManager

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager= CaptureManager(cv.VideoCapture(0),self._windowManager,True)
        
    def run(self):
        #run the loop
        self._windowManager.createWindow()
        while (self._windowManager.isWindowCreated()):
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            self._captureManager.exitFrame()
            self._windowManager.processEvent()
            
    def onKeypress(self,keycode):
        """ handle a keypress
        sapce  -> Take a screenshot
        tab    -> Start/Stop recording and screencast
        escape -> Quit
        """
        if (keycode == 32): # sapce
            self._captureManager.writeImage('screenshot.png')
        elif(keycode == 9): # tab
            if(not self._captureManager.isWrittingVideo):
                self._captureManager.startWrittingVideo('screencast.avi')
            else:
                self._captureManager.stopWrittingVideo()
        elif(keycode == 27): #escape
            self._windowManager.destroyWindow()