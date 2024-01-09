import pickle 
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
with open('coord_model.pkl','rb') as f:
    model=pickle.load(f)
cap =cv2.VideoCapture("male-Barbell-barbell-deadlift-front.avi")
counter=0
current_stage=''
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
landmarks=['class']
for val in range(1,33+1):
    landmarks+=['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=pose.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))   
        try:
            row=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X=pd.DataFrame([row],columns=landmarks[1:])
            body_language_class=model.predict(X)[0]
            body_language_prob=model.predict_proba(X)[0]
            print(body_language_class,body_language_prob)
            if body_language_class=='down' and body_language_prob[body_language_prob.argmmax()]>=0.7:
                current_stage='down'
            elif current_stage=='down' and  body_language_class=='up' and body_language_prob[body_language_prob.argmax()]>=0.7:
                current_stage='up'
                counter+=1
                print(counter)
                print(current_stage) 
                #get status box
            '''cv2.rectangle(image,(0,0),(225,60),(245,117,16),-1)
                #DISPALY CLASS
            cv2.putText(image,'CLASS',(95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,body_language_class.split(' ')[0],(90,40),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
                #DISPLAY PROB
            cv2.putText(image,'PROB',(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                #DISPLAY COUNT
            cv2.putText(image,'COUNT',(180,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,str(counter),(175,45),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)'''
        
        except Exception as e:
            pass
        cv2.rectangle(image,(0,0),(225,60),(245,117,16),-1)
                #DISPALY CLASS
        cv2.putText(image,'CLASS',(95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,body_language_class.split(' ')[0],(90,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                #DISPLAY PROB
        cv2.putText(image,'PROB',(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                #DISPLAY COUNT
        cv2.putText(image,'COUNT',(180,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(175,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA) 


        cv2.imshow('raw webcam counter',image)
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()    