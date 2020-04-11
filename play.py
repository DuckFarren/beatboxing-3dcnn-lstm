import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from keras.models import load_model
import random
from collections import deque
import threading
# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-n", "--num-frames", type=int, default=100,help="# of frames to loop over for FPS test")

model_path = "model\\3DCNN+3LSTM_128_4_aug_v4.h5"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
classes = ['Jap','Hook','Uppercut','None']
frames = deque(maxlen=24)
status = 0 # 0:start, 1:generate pose, 2:predict
# previous = 0

class OutputFrame:
    def __init__(self):
        self.frame = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3))
        self.label = 'Warming Up...'
        self.frameno = 0
        self.pose = ''
        self.score = 0
        self.message = ''
    
    def randomPose(self):
        global status
        if status == 1:
            index = random.randint(0,2)
            self.pose = classes[index]
            status = 2
    
    def checkPose(self,confidence):
        global status
        if self.pose == self.label:
            if confidence >= 0.75:
                self.message = 'PERFECT!'
                self.score += 100
            else:
                self.message = 'GOOD!'
                self.score += 50
        else:
            print('MISS')
            self.message = 'MISS!'
            status = 1
                
class WebcamThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    def run(self):
        print("Starting " + self.name)
        self.get_frame()
        print("Exiting " + self.name)
    def get_frame(self):
        while not done:
            _, frame = cap.read()
            output_frame.frame = frame

class PredictorThread(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
    def run(self):
        global model_path,status
        print("Starting " + self.name)
        print("[INFO] loading network...")
        self.model = load_model(model_path)
        print("[INFO] model loaded successfully...")
        status = 1
        self.predict()
        print("Exiting " + self.name)
    
    def predict(self):
        global frames, status
        while not done:
            _, image_np = cap.read()
            image_np = cv2.resize(image_np,(128,128),interpolation=cv2.INTER_AREA)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            frames.append(image_np)
            output_frame.frameno = len(frames)
            if len(frames) < 24:
                continue
            else:
                if status == 2:
                    output_frame.message = ''
                    np_frames = np.array(frames)
                    label, confidence = self.predict_label(np_frames)
                    for i in range(18):
                        frames.popleft()
                    print('pred: '+label)  
                    output_frame.label = label
                    output_frame.checkPose(confidence)
                    status = 1
                
    def predict_label(self, frames):  
        X_train = np.expand_dims(frames, axis=0)
        train_set = X_train.astype('float16')
        train_set -= 111.75
        train_set /= 143.2
        preds = self.model.predict(train_set)
        label = classes[np.argmax(preds,axis=1)[0]]
        confidence = np.max(preds,axis=1)[0]
        return label, confidence

if __name__ == "__main__":
    done = False

    cap = cv2.VideoCapture(0)
    cap.set(3, IMAGE_WIDTH)
    cap.set(4, IMAGE_HEIGHT)
    output_frame = OutputFrame()

    webcam_thread = WebcamThread("Webcam Thread")
    predictor_thread = PredictorThread("Predictor Thread")
    webcam_thread.start()
    predictor_thread.start()

    while True:
        to_show = output_frame.frame
        output_frame.randomPose()
        
        if status!=0:
            cv2.putText(to_show, str(output_frame.pose), (260, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
        cv2.putText(to_show, str(output_frame.message), (250,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, 8)
        cv2.putText(to_show, "Score: {}".format(output_frame.score), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(to_show, "Label: {}".format(output_frame.label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(to_show, str(output_frame.frameno), (580, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 100, 0), 4)
        cv2.imshow('frame', to_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True
            break

    cap.release()
    cv2.destroyAllWindows()