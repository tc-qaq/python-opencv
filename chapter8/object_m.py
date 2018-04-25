import cv2 as cv
import numpy as np
from pedestrian import Pedestrian

clicked = False;
def object_move_detect():
    camera = cv.VideoCapture(0)
    es     = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,4))
    kernel = np.ones((5,5),np.uint8)
    bg     = None
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        rect,frame = camera.read()
        # first frame as background
        if (bg is None):
            bg = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            bg = cv.GaussianBlur(bg,(21,21),0)
            continue
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray,(21,21),0)  #blur
        
        # compute the diff between bg and gray to get the difference map
        diff = cv.absdiff(bg,gray)        
        diff = cv.threshold(diff,25,255,cv.THRESH_BINARY)[1] # 2 value image        
        diff = cv.dilate(diff,es,2)                          # dilate 膨胀
        img, cnts, hire = cv.findContours(diff.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if (cv.contourArea(c) < 1500):
                continue
            (x,y,w,h) = cv.boundingRect(c)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv.imshow('contours',frame)
        cv.imshow('diff',diff)
    cv.destroyAllWindows()    
    camera.release()

def bg_split():
    camera = cv.VideoCapture(0)
    knn = cv.createBackgroundSubtractorKNN()
    mog = cv.createBackgroundSubtractorMOG2()
    
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    
    cv.setMouseCallback('MyWindow',onMouse)
    
    while (cv.waitKey(1) == -1 and not clicked):
        rect,frame = camera.read()
        fgmask = mog.apply(frame)
        fgmask1= knn.apply(frame)
        res    = abs(fgmask - fgmask1)
        cv.imshow('frame',fgmask)
        cv.imshow('frame1',fgmask1)
        cv.imshow('res',res)
    cv.destroyAllWindows()    
    camera.release()
    
def move_bg():
    bs = cv.createBackgroundSubtractorKNN(True)
    camera = cv.VideoCapture(0)
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        rect,frame = camera.read()
        fgmask = bs.apply(frame)
        thresh = cv.threshold(fgmask.copy(), 244,255, cv.THRESH_BINARY)[1]
        morph  = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        dilate = cv.dilate(thresh, morph, 2)
        img, cnts, hier = cv.findContours(dilate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if (cv.contourArea(c) > 1600):
                (x,y,w,h) = cv.boundingRect(c)
                cv.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),2)
        
        
        cv.imshow('mog',fgmask)
        cv.imshow('thresh',thresh)
        cv.imshow('detection',frame)
    cv.destroyAllWindows()    
    camera.release()
 
def MS():
    camera = cv.VideoCapture(0)
    ret,frame = camera.read()
    r,h,c,w = 10,200,10,200
    track_window = (c,r,w,h)
    roi = frame[r:r+h,c:c+w]
    hsv_roi = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi,np.array((100.,30.,32.)),np.array((180.,120.,255.)))
    
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])  # color hist
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        rect,frame = camera.read()
        if (rect == True):
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)  #possibly estimate
            rect,track_window = cv.meanShift(dst,track_window,term_crit)
            
            x,y,w,h = track_window
            img2    = cv.rectangle(frame,(x,y),(x+w,y+h),255,2)
            
            cv.imshow('Img',img2)
            
    cv.destroyAllWindows()    
    camera.release()
    
def CAMShift():    # CAMShift  (Continuously adaptive MeanShift)
    camera = cv.VideoCapture(0)
    ret,frame = camera.read()
    r,h,c,w = 300,200,400,300
    track_window = (c,r,w,h)
    
    roi = frame[r:r+h,c:c+w]
    hsv_roi = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi,np.array((150.,30.,32.)),np.array((180.,120.,255.)))
     
    roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])  # color hist
    cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        rect,frame = camera.read()
        if (rect == True):
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)  #possibly estimate
            rect,track_window = cv.CamShift(dst,track_window,term_crit)
            
            pts = cv.boxPoints(rect)
            pts = np.int0(pts)
            
            img2 = cv.polylines(frame,[pts],True,255,2) 
            
            cv.imshow('Img',img2)
            
    cv.destroyAllWindows()    
    camera.release()



frame = np.zeros((500,500,3),np.uint8)#Kalman + meanshift
last_measurement = current_measurement = np.array((2,1),np.float32)
last_prediction  = current_prediction  = np.zeros((2,1),np.float32)
def Kalman_filter_move_detect():  
    # mouse move event : draw mouse move and kalman predict  
    def mouseMove(event,x,y,s,p):
        global frame,last_measurement,current_measurement,last_prediction,\
               current_prediction
        last_prediction = current_prediction
        last_measurement= current_measurement
        
        current_measurement = np.array([[np.float32(x)],[np.float32(y)]])
        kalman.correct(current_measurement)
        #predict
        current_prediction = kalman.predict()
        lmx, lmy = last_measurement[0],last_measurement[1]
        cmx, cmy = current_measurement[0],current_measurement[1]
        lpx, lpy = last_prediction[0], last_prediction[1]
        cpx, cpy = current_prediction[0], current_prediction[1]
        #draw line
        cv.line(frame,(lmx,lmy),(cmx,cmy),(0,100,0))
        cv.line(frame,(lpx,lpy),(cpx,cpy),(0,0,200))
    
    cv.namedWindow('MyWindow')
    cv.setMouseCallback('MyWindow',mouseMove)
    
    kalman = cv.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                         [0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov   = np.array([[1,0,0,0],[0,1,0,0],
                                         [0,0,1,0],[0,0,0,1]],np.float32)
    while (cv.waitKey(1) == -1 ):
        cv.imshow('MyWindow',frame)
            
    cv.destroyAllWindows()    
      
def pedestrian_main():
    history = 20
    bs = cv.createBackgroundSubtractorKNN(True)
    bs.setHistory(history)
    cv.namedWindow('surveillance')
    pedestrians = {}
    first_frame = True
    frames = 0
    camera = cv.VideoCapture(0)
    while (cv.waitKey(1) == -1 ):
        grabbed, frame = camera.read()
        if(grabbed is False):
            break
        fgmask = bs.apply(frame)
        if( frames < history):
            frames += 1
            continue
        thresh = cv.threshold(fgmask.copy(), 127,255,cv.THRESH_BINARY)[1]
        thresh = cv.erode(thresh,
                          cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)),2)
        dilate = cv.dilate(thresh,
                          cv.getStructuringElement(cv.MORPH_ELLIPSE,(8,3)),2)
        
        img,cnts,hier = cv.findContours(dilate,
                                        cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cnt = 0
        for c in cnts:
            if(cv.contourArea(c) > 500):
                x,y,w,h = cv.boundingRect(c)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                if(first_frame == True):
                    pedestrians[cnt] = Pedestrian(cnt,frame,(x,y,w,h))
                cnt += 1 
        for i,p in pedestrians:
            p.update(frame)
        
        first_frame = False
        frames += 1
        
        cv.imshow('surveillance',frame)
    cv.destroyAllWindows()     
if __name__ == '__main__':
#     object_move_detect()
#     bg_split()
#     move_bg()
#     MS()
#     CAMShift()
#     Kalman_filter_move_detect()
    pedestrian_main()