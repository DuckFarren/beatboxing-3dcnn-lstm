import cv2
import time
import os

ds_path = 'datasets\\jap'
# ds_path = 'datasets\\hook'
# ds_path = 'datasets\\uppercut'
# ds_path = 'datasets\\none'
nfiles = len(os.listdir(ds_path))

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
capture_duration = 2
start_time = time.time()
file_name = ds_path+'\\'+ str(nfiles+1)+ '.avi'
writer = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'XVID'),30,(frame_width,frame_height))

while( int(time.time() - start_time) < capture_duration ):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
        writer.write(frame)
        cv2.imshow('frame',frame)
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows() 