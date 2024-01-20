import mediapipe as mp
import cv2
import numpy as np
import csv
import os
from matplotlib import pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
landmarks=['class']
for val in range(1,33+1):
    landmarks+=['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]
with open('Incline_db_coord.csv',mode='w',newline='') as f:
    csv_writer=csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)    
def export_landmarks(results,action):
    try:
        keypoints=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0,action)
        with open('Incline_db_coord.csv',mode='a',newline='') as f:
            csv_writer=csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass    
cap=cv2.VideoCapture('male-dumbbell-incline-bench-press-front_q2q0T12.avi')
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        #make Detection
        results=pose.process(image)
        #recolor immage back to BGR for rendering
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS
                                  ,mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))
        k=cv2.waitKey(1)
        if k==117:
            export_landmarks(results,'up')
        if k==100:
            export_landmarks(results,'down')
        cv2.imshow('Raw Webcam Feed',image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()                