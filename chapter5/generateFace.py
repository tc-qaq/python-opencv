import cv2 as cv
import numpy as np



clicked = False;
def generate():
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    eye_cascade  = cv.CascadeClassifier('../chapter5/cascades/haarcascade_eye.xml')
    
    camera = cv.VideoCapture(0)
    count = 0
    cv.namedWindow('MyWindow')
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        ret, frame = camera.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray,1.3,5) 
        for (x,y,w,h) in faces:
            img = cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            f   = cv.resize(gray[y:y+h,x:x+w],(200,200))
            cv.imwrite('../images/at/%s.pgm' % str(count),f)
            count += 1
        cv.imshow("MyWindow",f)

    camera.release()
    cv.destroyAllWindows()
def writeCSV():
    fw = open('../images/at/files.csv','w')
    for i in range(141):
        fw.write('../image/at/' + str(i)+'.pgm\n');
    fw.close()
def read_imges(files):
    c = 0
    x,y = [], []
    fr = open(files,'r')
    paths = []
    for line in fr.readlines():
        item = line.strip().split(',')[0]
        print(item)
        paths.append(item)
    for path in paths:
        img = cv.imread(path)
        x.append(np.asarray(img,dtype=np.uint8))
        y.append(0)
        c += 1
    return [x,y]
    
def face_rec():
    names = ['tc']
    [x,y] = read_imges('../images/at/files.csv')
    y     = np.asarray(y, dtype=np.int32)
    outdir= '../images/at'
    model = cv.face.EigenFaceRecognizer_create()
    model.train(np.asarray(x),np.asarray(y))
    camera = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    
    def onMouse(event,x,y,flags,param): 
        global clicked
        if (event == cv.EVENT_LBUTTONUP):
            clicked = True 
    cv.namedWindow('MyWindow')
    cv.setMouseCallback('MyWindow',onMouse)
    while (cv.waitKey(1) == -1 and not clicked):
        ret, img = camera.read()
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)        
        faces = face_cascade.detectMultiScale(gray,1.3,5) 
        for (x,y,w,h) in faces:
            img = cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi = cv.resize(gray[y:y+h,x:x+w],(200,200))
            try:
                params = model.predict(roi)
                cv.putText(img,names[params[0]],(x,y-20),cv.FONT_HERSHEY_SIMPLEX,1,25,2)
            except:
                continue
        cv.imshow("MyWindow",img)

    camera.release()
    cv.destroyAllWindows()
        
    
    
    
if __name__ == '__main__':
    #generate()
    #writeCSV()
    face_rec()