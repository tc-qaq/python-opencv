import cv2 as cv
import numpy as np
from random import randint 
from numpy import dtype

def simple_ann():
    ann = cv.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([9,5,9],dtype=np.uint8))
    ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
    
    ann.train(np.array([[1.2,1.3,1.9,2.2,2.3,2.9,3.0,3.2,3.3]], dtype=np.float32),
              cv.ml.ROW_SAMPLE,
              np.array([[0,0,0,0,0,1,0,0,0]], dtype=np.float32))
    print(ann.predict(np.array([[1.4,1.5,1.2,2.,2.5,2.8,3.,3.1,3.8]],dtype=np.float32)))
    
def animal_ann():
    animal_net = cv.ml.ANN_MLP_create()
    animal_net.setTrainMethod(cv.ml.ANN_MLP_BACKPROP|cv.ml.ANN_MLP_UPDATE_WEIGHTS)
    animal_net.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM)
    animal_net.setLayerSizes(np.array([3,8,4]))
    animal_net.setTermCriteria((cv.TERM_CRITERIA_EPS|cv.TermCriteria_COUNT,10,1))
    
    def dog_sample():
        return [randint(5,20),1,randint(38,42)]
    def dog_class():
        return [1,0,0,0]
    
    def condor_sample():
        return [randint(3,13),3,0]
    def condor_class():
        return [0,1,0,0]
    
    def dolphin_sample():
        return [randint(30,190),randint(5,15),randint(80,100)]
    def dolphin_class():
        return [0,0,1,0]
    
    def dragon_sample():
        return [randint(1200,1800),randint(15,40),randint(110,180)]
    def dragon_class():
        return [0,0,0,1]
    
    def record(sample, c):
        return (np.array([sample],dtype=np.float32),np.array([c],dtype=np.float32))
    
    records = []
    samples = 5000
    for x in range(samples):
        records.append(record(dog_sample(),dog_class()))
        records.append(record(condor_sample(),condor_class()))
        records.append(record(dolphin_sample(),dolphin_class()))
        records.append(record(dragon_sample(),dragon_class()))
    
    for n in range(5):
        for t, c in records:
            animal_net.train(t, cv.ml.ROW_SAMPLE, c)
        
    print(animal_net.predict(np.array([dog_sample()],dtype=np.float32)))
    print(animal_net.predict(np.array([condor_sample()],dtype=np.float32)))
    print(animal_net.predict(np.array([dolphin_sample()],dtype=np.float32)))
    print(animal_net.predict(np.array([dragon_sample()],dtype=np.float32)))

if __name__ == '__main__':
#     simple_ann()
    animal_ann()