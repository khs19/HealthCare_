import pickle
import numpy as np
import argparse
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

ap = argparse.ArgumentParser()
c = {}

ap.add_argument("-a", "--activity", required=True,
	help="activity to be recorder")
ap.add_argument("-v", "--video", required=True,
	help="video file from which keypoints are to be extracted")
ap.add_argument("-l", "--lookup", default="lookup_new.pickle",
	help="The pickle file to dump the lookup table")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])
cnt = 0
total_list = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    if not cap.isOpened():
        print('error')
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret == False: #file's end
            print('endFile')
            break
        
        # 이미지를 다시 RGB형식으로 칠함 (먼저는 프레임을 잡아줘야한다)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # 이미지 다시쓰기
        
        # 탐지하기
        results = pose.process(image)
        cnt += 1
        
        if results.pose_landmarks is not None:
            part_list = []
            for i in range(33):
                part_list.append(results.pose_landmarks.landmark[i].x)
                part_list.append(results.pose_landmarks.landmark[i].y)
                part_list.append(results.pose_landmarks.landmark[i].z)
        else:
            continue    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        total_list.append(np.array(part_list).reshape(33, 3))
    
    total_list = np.array(total_list)
    
    cap.release()
    cv2.destroyAllWindows()
    
    c[args["activity"]] = total_list
    f = open(args["lookup"],'wb')
    pickle.dump(c,f)
    
#use method
#python pickle_file_create.py -a exercise_name -l lookuptable_path -v video_path