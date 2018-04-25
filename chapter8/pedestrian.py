import cv2 as cv
import numpy as np
import argparse 
from numpy.core.defchararray import center

parser = argparse.ArgumentParser()
parser.add_argument('-alg', default = '', help = 'algorithm')
args = parser.parse_args()


class Pedestrian:
    def __init__(self,id, frame, track_window):
        self.id = int(id)
        x,y,w,h = track_window
        self.track_window = track_window
        self.roi = cv.cvtColor(frame[y:y+h,x:x+w],cv.COLOR_BGR2HSV)
        roi_hist = cv.calcHist([self.roi],[0],None,[16],[0,180])
        self.roi_hist = cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
        
        self.kalman = cv.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                                  [0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov   = np.array([[1,0,0,0],[0,1,0,0],
                                                  [0,0,1,0],[0,0,0,1]],np.float32)
        self.measurement = np.array((2,1),np.float32)
        self.prediction  = np.array((2,1),np.float32)
        self.term_crit   = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)
        
    def __del__(self):
        print( "Pedestrian %d destroyed" % self.id)
    
    def update(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        bp  = cv.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        
        if(args.alg == 'c'):
            rect, self.track_window = cv.CamShift(bp, self.track_window,
                                                  self.term_crit)
            pts = cv.boxPoints(rect)
            pts = np.int0(pts)
            self.center = center(pts)
            cv.polylines(frame, [pts], True, 255, 1)
        
        if (args.alg == '' or args.alg == 'm'):
            rect, self.track_window = cv.meanShift(bp, self.track_window,
                                                  self.term_crit)
            x,y,w,h = self.track_window
            self.center = center([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
            cv.rectangle(frame, [x,y], [x+w,y+h],(255,255,0),1)
        
        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        cv.circle(frame, (int(prediction[0]),int(prediction[1])),4,(0,255,0),-1)
        font = cv.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv.putText(frame,"ID: %d-->%s"%(self.id,self.center),
                   (11,(self.id+1)*25+1),font,0.6,(0,0,0),1,cv.LINE_AA)
        cv.putText(frame,"ID: %d-->%s"%(self.id,self.center),
                   (10,(self.id+1)*25),font,0.6,(0,255,0),1,cv.LINE_AA) 
        