import cv2 as cv


def detect(filename):
    face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
    eye_cascade  = cv.CascadeClassifier('../chapter5/cascades/haarcascade_eye.xml')
    pro_cascade  = cv.CascadeClassifier('../chapter5/cascades/haarcascade_profileface.xml')
    img = cv.imread(filename)
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
    cv.imshow('face',img)
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    filename = '../images/face4.png'    
    detect(filename)    