import sys
import mediapipe as mp
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QCursor
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5.QtCore import *
import cv2
import threading
import time
from fileRead import file
from pickle_file_load import load
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw_ndim
from multiprocessing import Process, Value, Lock, Queue, Manager

# UI 파일 연결
form_class = uic.loadUiType("ui/main_ui.ui")[0]
form_start = uic.loadUiType("ui/start_ui.ui")[0]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 256, 0))

mp_pose = mp.solutions.pose

dataset_dir = './dataset'
file_ = file(dataset_dir)
all_file = sorted(file_.find_all_file_paths())

exercise_queue = Queue(1)
user_queue = Queue(1)

score_value = Value('i', 0)
mean = Value('i', 0)
next_video_value = Value('i', 0)
replay_value = Value('i', 0)
exit_value = 0
exercise_keypoint_list = []


def process(exercise_keypoint_q: Queue(), user_keypoint_q: Queue(), share_score: Value, share_replay: Value, share_next: Value):
    scoreList = []  # 점수의 경향을 보는 list
    while True:
        if exercise_keypoint_q.empty() or user_keypoint_q.empty():  # 둘중 하나의 q라도 비어있으면 비교를 진행하지 않는다.
            continue

        user_keypoint_list = user_keypoint_q.get()
        exercise_keypoint_list = exercise_keypoint_q.get()

        if share_next.value == 1 or share_replay.value == 1:
            continue

        if user_keypoint_list.shape[0] >= 20 and exercise_keypoint_list.shape[0] > 0:
            similarity = cosine_similarity(exercise_keypoint_list[0: 20].reshape(1, -1),user_keypoint_list[0: 20].reshape(1, -1)).squeeze()  # similarity 계산
            if similarity < 0.95:  # 동작의 앞 부분이 0.95를 넘지 못한다면
                share_replay.value = 1  # 사용자 영상 keypoint 초기화후 재시작
                scoreList = []
            else:
                if 20 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] + 50 and share_replay.value != 1 and share_next.value != 1:
                    x = dtw_ndim.distance(user_keypoint_list, exercise_keypoint_list)
                    x = (max(((x - 15) / 12), 0) - (1 / 2)) * 7
                    x = 1 / (1 + np.exp(x))
                    score = int(x * 100)  # 점수 계산
                    share_score.value = score  # 점수를 다른 프로세스도 볼 수 있도록 업데이트 why? --> set_score하는 부분은 User Class에서 진행되기 때문 !
                    
                    scoreList.append(score)

                    if (exercise_keypoint_list.shape[0] / 1.3 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] + 50) and score < 10:
                        #IO process로부터 입력된 사용자 영상의 길이가 영상 길이와 비교했을 때 충분하고, 점수가 10점 이하일 때 지금까지 들어왔던 사용자 영상 초기화
                        scoreList = []  # 점수 기록 초기화
                        share_replay.value = 1  # 사용자 영상 재 수집 요청
                        
                    elif exercise_keypoint_list.shape[0] / 1.5 < user_keypoint_list.shape[0] and np.array(scoreList).mean() >= score + 5:
                        #IO process로부터 입력된 사용자 영상의 길이가 영상 길이와 비교했을 때 충분하고, 점수가 이전 점수들에 비해 떨어지는 경향을 보이는 경우 사용자 영상 초기화
                        scoreList = []  # 점수 기록 초기화
                        share_replay.value = 1  # 재시작

                    elif score >= 80 and (exercise_keypoint_list.shape[0] / 1.3 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] * 1.3):
                        #IO process로부터 입력된 사용자 영상의 길이가 영상 길이와 비교했을 때 충분하고, 점수가 80점 이상일 때 다음영상으로 넘어감
                        scoreList = []  # 점수 기록 초기화
                        share_next.value = 1  # 사용자 영상 재 수집 및 다음 영상 요청

        if user_keypoint_list.shape[0] >= exercise_keypoint_list.shape[0] + 50:  # 사용자가 너무 느리게 따라하는 경우
            scoreList = []
            share_replay.value = 1

class Video(QThread):
    update_video_frame_signal = pyqtSignal(QtGui.QImage)
    set_label_signal = pyqtSignal(str, str)
    show_dlg_signal = pyqtSignal(str, str, str, str)
    exit_dlg_signal = pyqtSignal(str, str)
    kill_thread_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        threading.Thread.__init__(self)
        self.pickle_dir = './coordinate_pickle/lookup_'  # pickle file의 root dir

    def run(self):
        global exercise_keypoint_list
        global next_video_value
        FPS = 40  # video frame setting
        counter = 0
        for file_name in all_file:  # 모든 파일들 가져와서 순서대로
            counter += 1
            exercise = file_name.split('/')[2]  # ['.', 'root_dir', 'level_name', 'train or test.mp4']
            exercise_level = exercise.split('_')[0]  # level
            exercise_name = exercise.split('_')[1]  # name

            # 운동 이름과 난이도 setText 추가
            self.set_label_signal.emit(exercise_name, exercise_level)

            pickle_dir = self.pickle_dir + exercise_name + '.pickle'  # Find pickleFile dir

            exercise_keypoint_list = load(lookupfile_path=pickle_dir, activity_name=exercise_name).array()  # pickle file loaded

            cap = cv2.VideoCapture(file_name)
            while cap.isOpened():                
                if exit_value == 1:  # ESC나 종료 버튼 클릭
                    self.exit_dlg_signal.emit(str(0), str(0))
                    time.sleep(100)
                    #self.kill_thread_signal.emit()

                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):  # 영상의 현재 프레임이 영상의 최대 프레임(끝)과 같을때, 즉 영상의 끝일 때
                    if len(exercise_keypoint_list) == 0:  # 영상 list안이 비어있을 때  User Class에서 exercise_keypoint_list를 비워버렸을 때 다음 영상으로 넘어가야한다.
                        last_score = str(score_value.value)
                        score_value.value = 0  # Score를 0으로 setting
                        print('next')
                        print(counter)
                        if counter == len(all_file):
                            self.exit_dlg_signal.emit(all_file[counter-1].split('/')[2].split('_')[1], last_score)
                            print("end")
                            time.sleep(100)
                            #self.kill_thread_signal.emit()
                        else:
                            self.show_dlg_signal.emit(all_file[counter].split('/')[2].split('_')[1], exercise_level, last_score, all_file[counter-1].split('/')[2].split('_')[1])
                        time.sleep(3)
                        next_video_value.value = 0
                        break

                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video replay

                _, frame = cap.read()
                # 이미지를 다시 RGB형식으로 칠함 (먼저는 프레임을 잡아줘야한다)
                frame = cv2.flip(frame, 1)  # 양수: 좌우 대칭
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  # 이미지 다시쓰기

                image = cv2.resize(image, (800, 900), interpolation=cv2.INTER_CUBIC)

                time.sleep(1 / FPS)  # Adjust frame speed

                h, w, c = image.shape  # height, weight, channel
                qImg = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
                self.update_video_frame_signal.emit(qImg)  # update video frame

            cap.release()  # free memory


class User(QThread):
    update_user_frame_signal = pyqtSignal(QtGui.QImage)
    set_score_signal = pyqtSignal(str)
    cam_not_open_signal = pyqtSignal()
    set_start_text_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        threading.Thread.__init__(self)

    def run(self):
        global exercise_queue
        global user_queue
        global score_value
        global replay_value
        global next_video_value
        global exercise_keypoint_list

        cam = cv2.VideoCapture(0)  # VideoCapture(index): index는 카메라 장치 번호 의미, 웹캠: 0
        user_keypoint_list = []
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            if not cam.isOpened():
                print("카메라가 켜져있지 않습니다.")
                exit()
            else:  # camera is open
                while cam.isOpened():
                    if score_value.value == 0: # 0점일 때 '운동을 시작해주세요' 띄워야한다~
                        self.set_start_text_signal.emit(1)
                    else:
                        self.set_start_text_signal.emit(0)
                        
                    status, frame = cam.read()  # 카메라의 상태 및 프레임 받아옴, 정상 작동일 경우 status = True

                    if status:
                        self.set_score_signal.emit(str(score_value.value))  # score 업데이트
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색상 공간 변환 함수
                        results = pose.process(frame)
                        frame.flags.writeable = True
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec)

                        frame = cv2.resize(frame, dsize=(800, 900), interpolation=cv2.INTER_CUBIC)
                        frame = cv2.flip(frame, 1)  # 양수: 좌우 대칭

                        h, w, c = frame.shape  # height, weight, channel
                        qImg = QImage(frame.data, w, h, w * c, QImage.Format_RGB888)
                        self.update_user_frame_signal.emit(qImg)

                        if replay_value.value >= 1 or next_video_value.value >= 1:  # replay, next 변수가 변경되어 있을 땐 keypoint 를 queue에 넣지 않는다
                            if replay_value.value >= 1:  # replay_value가 1이라면 list를 처음부터 다시 추출한다.
                                user_keypoint_list = []
                                replay_value.value = 0
                                score_value.value = 0

                            elif next_video_value.value >= 1:
                                user_keypoint_list = []
                                exercise_keypoint_list = []
                                # next_video_value.value = 0
                            continue

                        if len(exercise_keypoint_list) == 0:  # exercise keypoint is empty
                            user_keypoint_list = []
                            continue

                        if results.pose_landmarks is not None:
                            landmark = results.pose_landmarks.landmark
                            one_frame_keypoint_list = []
                            for i in range(33):  # keypoint num is 33
                                one_frame_keypoint_list.append(landmark[i].x)
                                one_frame_keypoint_list.append(landmark[i].y)
                                one_frame_keypoint_list.append(landmark[i].z)

                            if np.array(one_frame_keypoint_list).shape == (99,):  # 중간에 검출이 안된 키포인트가 없을때
                                # print('중간에 검출이 안된 키포인트가 없을때')
                                user_keypoint_list.append(np.array(one_frame_keypoint_list).reshape(33, 3))

                            else:  # 영상중간에 키포인트들이 검출이 안된다? 다시 시작 !
                                score_value.value = 0
                                user_keypoint_list = []

                            if np.array(user_keypoint_list).shape[0] >= 30 and np.array(exercise_keypoint_list).shape[0] > 0:  # 둘다 길이가 일정 이상일 때
                                if user_queue.qsize() == 0 and exercise_queue.qsize() == 0:
                                    user_queue.put(np.array(user_keypoint_list))  # 프로세스간 공유 자원 큐에 list를 넣는다.
                                    exercise_queue.put(np.array(exercise_keypoint_list))

                        else:  # pose landmark가 보이지 않는다면 지금까지 추출되었던 list 초기화
                            user_keypoint_list = []
                            # print('pose landmark가 보이지 않는다면 지금까지 추출되었던 list 초기화')
                            score_value.value = 0

                    else:
                        self.cam_not_open_signal.emit()
                        break

                cam.release()  # free memory


# 화면을 띄우는데 사용되는 Class 선언
class Window(QMainWindow, form_class):
    counter = 0
    score_list = []

    global exercise_queue
    global user_queue
    global score_value
    global next_video_value
    global replay_value

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.dlg_window = DialogWindow()
        self.exit_dlg_window = ExitDialogWindow()

        # exercise video thread 생성
        self.ex_th = Video()
        self.ex_th.update_video_frame_signal.connect(self.update_video_frame_slot)
        self.ex_th.set_label_signal.connect(self.set_label_slot)
        self.ex_th.show_dlg_signal.connect(self.show_dlg_slot)
        self.ex_th.exit_dlg_signal.connect(self.exit_dlg_slot)
        self.ex_th.kill_thread_signal.connect(self.kill_thread)
        self.ex_th.start()

        # user cam thread 생성
        self.user_th = User()
        self.user_th.update_user_frame_signal.connect(self.update_user_frame_slot)
        self.user_th.set_score_signal.connect(self.set_score_slot)
        self.user_th.cam_not_open_signal.connect(self.cam_not_open_slot)
        self.user_th.set_start_text_signal.connect(self.set_start_text_slot)
        self.user_th.start()        

        # 버튼 클릭 이벤트 함수 연결 - 종료하기
        self.exit_button.clicked.connect(self.exit)
        self.exit_dlg_window.exit_button.clicked.connect(self.kill_thread)
        self.proc = Process(target=process, args=(exercise_queue, user_queue, score_value, replay_value, next_video_value))
        self.proc.start()

    def exit(self):
        global exit_value
        exit_value = 1

    @pyqtSlot(str, str)
    def set_label_slot(self, name, level):
        self.exercise_name.setText(name)
        self.exercise_level.setText(level)
    
    @pyqtSlot(int)
    def set_start_text_slot(self, isStart):
        if isStart:
            self.exercise_start_text.setText("운동을 시작해주세요")
            self.exercise_start_text.setStyleSheet("background-color : white;" 
                                       "color:black;"
                                       "font-weight:600;"
                                       "border-radius:20px")
        else:
            self.exercise_start_text.setText("운동중입니다")
            self.exercise_start_text.setStyleSheet("background-color : #3c4043;" 
                                       "color:white;")
        
    
    @pyqtSlot(str)
    def set_score_slot(self, score):
        self.exercise_score.setText(score)

    @pyqtSlot(QImage)
    def update_user_frame_slot(self, image):
        self.user_cam_screen.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def cam_not_open_slot(self):
        self.user_cam_screen.setText("카메라가 연결되어 있지 않습니다.")

    @pyqtSlot(QImage)
    def update_video_frame_slot(self, image):
        self.exercise_screen.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str, str, str, str)
    def show_dlg_slot(self, next_ex_name, passed_ex_level, passed_ex_score, passed_ex_name):
        self.counter += 1
        self.score_list.append(float(passed_ex_score))
        self.exit_dlg_window.addText(passed_ex_name, passed_ex_score)
        self.dlg_window.set_dialog_text(passed_ex_level, passed_ex_score, next_ex_name)
        self.dlg_window.show()
        self.dlg_window.start_timer()

    @pyqtSlot(str, str)
    def exit_dlg_slot(self, last_exercise, last_score):
        if self.counter == 0:
            self.exit_dlg_window.addText("진행한 운동 없음", str(0))
        else:
            self.score_list.append(float(last_score))
            self.counter += 1
            self.exit_dlg_window.addText(last_exercise, last_score)
            self.exit_dlg_window.set_result_score(round((sum(self.score_list)/self.counter), 2))
            
        self.exit_dlg_window.set_exit_text()
        self.exit_dlg_window.show()

    # @pyqtSlot()
    def kill_thread(self):
        self.user_th.quit()  # thread 종료
        self.ex_th.quit()
        self.proc.terminate()
        self.exit_dlg_window.exit_dlg_window_close()
        self.close()
        exit(0)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:  # ESC 누르면 종료
            global exit_value
            exit_value = 1

        if e.key() == Qt.Key_N:
            next_video_value.value = 1

class startWindow(QWidget, form_start):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        pixmap = QPixmap('ui/logo.png')
        self.logo.setPixmap(pixmap)

        self.start_button.clicked.connect(self.btn_start_to_main)

    def btn_start_to_main(self):
        self.hide()  # 시작화면 숨김
        time.sleep(1)
        self.main = Window()
        self.main.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:  # ESC 누르면 종료
            exit(0)


class DialogWindow(QDialog):
    def __init__(self): 
        super().__init__()
        dialog_ui = 'ui/next_video_dialog.ui'
        uic.loadUi(dialog_ui, self)

        self.sec = 3
        self.my_timer = QTimer(self)
        self.my_timer.timeout.connect(self.timer_timeout)
        self.timer_second.setText(str(self.sec))
        self.center()

    def start_timer(self):
        self.my_timer.setInterval(1000)
        self.my_timer.start()  # 1초 마다 timer_timeout 함수 실행

    def timer_timeout(self):
        self.sec -= 1
        self.timer_second.setText(str(self.sec))  # 몇 초 남았는지 set
        if self.sec == 0:
            self.my_timer.stop()
            self.sec = 3
            self.timer_second.setText(str(self.sec))
            self.close()

    def set_dialog_text(self, level, score, next_ex_name):
        # 통과한 레벨 출력
        self.passed_level.setText(f":  {level}")
        # 평균 정확도 출력
        self.passed_score.setText(f":  {score}")
        # 다음 운동 이름 출력
        self.next_exercise.setText(f":  {next_ex_name}")

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class ExitDialogWindow(QDialog):
    def __init__(self):
        super().__init__()
        dialog_ui = 'ui/exit_dialog.ui'
        uic.loadUi(dialog_ui, self)

        self.exercise_name_list = ""
        self.exercise_score_list = ""
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def addText(self, ex_name, ex_score):
        self.exercise_name_list += ex_name + '\n\n'
        self.exercise_score_list += ex_score + '\n\n'

    def set_exit_text(self):
        self.result_name_list.setText(self.exercise_name_list)
        self.result_score_list.setText(self.exercise_score_list)

    def set_result_score(self, result):
        self.result_score.setText(str(result))

    def exit_dlg_window_close(self):
        self.close()

if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # Window Class의 인스턴스 생성
    startWindow = startWindow()
    startWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    sys.exit(app.exec_())
