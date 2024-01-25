import mediapipe as mp
import cv2
import numpy as np
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33 + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# Open CSV file in 'a+' mode
with open('sumo_coord.csv', mode='a+', newline='') as f:
    # Set the file cursor at the end of the file
    f.seek(0, os.SEEK_END)

    # If the file is empty, write the header
    if f.tell() == 0:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

def export_landmarks(results, action):
    try:
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0, action)
        with open('sumo_coord.csv', mode='a+', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass

cap = cv2.VideoCapture('male-Barbell-barbell-sumo-deadlift-front.avi')
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detection
        results = pose.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        k = cv2.waitKey(1)
        if k == 122:
            export_landmarks(results, 'up')
        if k == 100:
            export_landmarks(results, 'down')

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
