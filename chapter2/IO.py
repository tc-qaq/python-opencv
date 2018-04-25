#utf-8
import cv2 as cv
import numpy as np
import os
from numpy import reshape
from cameo import Cameo


#read and write
def r_and_w():
    # simple np
    img = np.zeros((3,3),dtype=np.uint8)
    print(img.shape,img)
    img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    print(img.shape, img)
    # read
    img = cv.imread("../images/lena.png")
    img1= cv.imread("../images/airplane.png",cv.IMREAD_GRAYSCALE)
    cv.imwrite("../images/airplaneGray.png",img1)
    # random array
    randomByteArray = bytearray(os.urandom(120000))
    flatNumpyArray  = np.array(randomByteArray)
    grayImage = flatNumpyArray.reshape(300,400)
    bgrIamge  = flatNumpyArray.reshape(100,400,3)
    
    # print item
    print('------item operator------')
    print( img.item(0,0,0) )
    img.itemset((0,0,0),255)
    print( img.item(0,0,0) )
    
    print(img.shape,img.size,img.dtype);
    
    
    #创建窗口并显示图像
    bgrImg = np.array(img)
    print(bgrImg)
    cv.namedWindow("Image")
    cv.imshow("Image",img)
    cv.imshow("GrayImage",grayImage)
    cv.imshow("BgrImage",bgrIamge)
    cv.waitKey(0)
    #释放窗口
    cv.destroyAllWindows() 
    
def video_rw():
    # video
    videoCapture = cv.VideoCapture('../card.mp4')
    fps  = videoCapture.get(cv.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))) 
    print(fps,size);
    # fourcc('I','4','2','0') too big avi
    # fourcc('P','I','M','1') MPEG-1 avi
    # fourcc('X','V','I','D') MPEG-2 avi
    # fourcc('T','H','E','O') Ogg Vorbis ogv
    # fourcc('F','L','V','1') Flash flv
    videoWrite = cv.VideoWriter('../output.avi',cv.VideoWriter_fourcc('X','V','I','D'),fps,size)
    success,fram = videoCapture.read()
    while success:
        videoWrite.write(fram)
        success,fram = videoCapture.read()
        
def video_Cap():
    # video
    videoCapture = cv.VideoCapture(0)
    fps = 30;
    size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))) 
    print(size)
    videoWrite = cv.VideoWriter('../output.avi',cv.VideoWriter_fourcc('X','V','I','D'),fps,size)
    success,frame = videoCapture.read()
    numFramesRemaining = 10*fps -1
    while success and numFramesRemaining>0:
        videoWrite.write(frame)
        success,frame = videoCapture.read()
        numFramesRemaining -=1
    videoCapture.release()
    
clicked = False;
def video_Cap_show():        
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True  
    def segment(img):        
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
        return img
    def detect(img):
        face_cascade = cv.CascadeClassifier('../chapter5/cascades/haarcascade_frontalface_default.xml')
        eye_cascade  = cv.CascadeClassifier('../chapter5/cascades/haarcascade_eye.xml')
        pro_cascade  = cv.CascadeClassifier('../chapter5/cascades/haarcascade_profileface.xml')
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        eyes  = eye_cascade.detectMultiScale(gray,1.3,5)
        pros  = pro_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        for (x,y,w,h) in eyes:
            img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        for (x,y,w,h) in pros:
            img = cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
        return img
    videoCapture = cv.VideoCapture(0)
    cv.namedWindow('MyWindow')
    cv.setMouseCallback('MyWindow',onMouse)
    print('SHowing camera feed Click window or press any key to stop')    
    success,frame = videoCapture.read()    
    while success and cv.waitKey(1) == -1 and not clicked:
        frame = detect(frame)
        cv.imshow('MyWindow',frame)
        success,frame = videoCapture.read()
    cv.destroyAllWindows()   
    videoCapture.release()


if __name__ == "__main__":
    #r_and_w()
    #video_rw()
    #video_Cap()
    video_Cap_show()
    """ handle a keypress
    sapce  -> Take a screenshot
    tab    -> Start/Stop recording and screencast
    escape -> Quit
    """
    #Cameo().run()