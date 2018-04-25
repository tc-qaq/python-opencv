import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

#HOG (histogram of oriented gradient 梯度直方图)
#split the image to multi segment as hostogram
# has problems in:     scale and locate
# solutions for above: image pyramid and sliding window 
def HOG():
    def is_inside(o, i):
        ox, oy, ow, oh = o
        ix, iy, iw, ih = i
        return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih
    
    def draw_person(image, person):
      x, y, w, h = person
      cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    img = cv.pyrDown(cv.imread("../images/people.jpg"))  
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    
    found, w = hog.detectMultiScale(img, winStride=(8,8),scale=1.05)
    
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    for person in found_filtered:
      draw_person(img, person)
    
    cv.imshow("people detection", img)  
    cv.waitKey(0)
    cv.destroyAllWindows()

 
def CarDetect():
    datapath = '../images/CarData/TrainImages'
    def path(cls,i):
        return "%s/%s%d.pgm" % (datapath,cls,i+1)
    
    pos, neg = "pos-" , "neg-"
    detect  = cv.xfeatures2d.SIFT_create()
    extract = cv.xfeatures2d.SIFT_create()
    
    flann_params = dict(algorithm=1,trees=5)
    flann = cv.FlannBasedMatcher(flann_params,{})
    
    bow_kmeans_trainer = cv.BOWKMeansTrainer(40)
    extract_bow = cv.BOWImgDescriptorExtractor(extract,flann)
    
    def extract_sift(fn):
        im = cv.imread(fn,0)
        return extract.compute(im,detect.detect(im))[1]
    for i in range(10):
        print(path(pos,i),path(neg,i))
        bow_kmeans_trainer.add(extract_sift(path(pos,i)))
        bow_kmeans_trainer.add(extract_sift(path(neg,i)))
    
    voc = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(voc)
    
    def bow_features(fn):
        im = cv.imread(fn,0)
        return extract_bow.compute(im, detect.detect(im))
    
    traindata, trainlabels = [],[]
    for i in range(20):
        print(path(pos,i))
        traindata.extend(bow_features(path(pos,i)))
        trainlabels.append(1)
        traindata.extend(bow_features(path(neg,i)))
        trainlabels.append(-1)
    
    svm = cv.ml.SVM_create()
    svm.train(np.array(traindata), cv.ml.ROW_SAMPLE, np.array(trainlabels))
    
    def predict(fn):
        f = bow_features(fn)
        p = svm.predict(f)
        print(fn,'\t',p[1][0][0])
        return p
    
    car,notcar = '../images/car1.jpg','../images/airplane.png'
    car_img    = cv.imread(car)
    notcar_img = cv.imread(notcar)
    car_predict= predict(car)
    notcar_predict = predict(notcar)
    
    font = cv.FONT_HERSHEY_COMPLEX
    
    if(car_predict[1][0][0] == 1.0):
        cv.putText(car_img,'Car Detected',(10,30),font,1,(0,255,0),2,cv.LINE_AA)
    if(notcar_predict[1][0][0] == -1.0):
        cv.putText(notcar_img,'Car Not Detected',(10,30),font,1,(0,255,0),2,cv.LINE_AA)
    
    cv.imshow('detect',car_img)
    cv.imshow('not',notcar_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
#     HOG()
    CarDetect()