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
form_class = uic.loadUiType("ui/mainUI.ui")[0]
form_start = uic.loadUiType("ui/start_screen.ui")[0]

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


def process(exercise_keypoint_q: Queue(), user_keypoint_q: Queue(), share_score: Value, share_replay: Value,
            share_next: Value):
    scoreList = []  # 점수의 경향을 보는 list
    while True:
        if exercise_keypoint_q.empty() or user_keypoint_q.empty():  # 둘중 하나의 q라도 비어있으면 비교를 진행하지 않는다.
            # print('queue가 비어있습니다.')
            continue

        user_keypoint_list = user_keypoint_q.get()
        exercise_keypoint_list = exercise_keypoint_q.get()

        if share_next.value == 1 or share_replay.value == 1:
            continue

        if user_keypoint_list.shape[0] >= 20 and exercise_keypoint_list.shape[0] > 0:
            similarity = cosine_similarity(exercise_keypoint_list[0: 20].reshape(1, -1),user_keypoint_list[0: 20].reshape(1, -1)).squeeze()  # similarity 계산
            if similarity < 0.95:  # 동작의 앞 부분이 0.95를 넘지 못한다면
                share_replay.value = 1  # 사용자 영상 keypoint초기화후 재시작
                scoreList = []
            else:
                if 20 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] + 50 and share_replay.value != 1 and share_next.value != 1:
                    x = dtw_ndim.distance(user_keypoint_list, exercise_keypoint_list)
                    x = (max(((x - 15) / 12), 0) - (1 / 2)) * 7
                    x = 1 / (1 + np.exp(x))
                    score = int(x * 100)
                    # score = int(100 - min(100, x))  # 점수 계산
                    share_score.value = score  # 점수를 다른 프로세스도 볼 수 있도록 업데이트 why? --> setScore하는 부분은 User Class에서 진행되기 때문 !
                    
                    scoreList.append(score)

                    if (exercise_keypoint_list.shape[0] / 1.3 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] + 50) and score < 10:
                        scoreList = []  # 점수 기록 초기화
                        share_replay.value = 1  # 사용자 영상 재 수집 요청
                        print('score 10')
                        print('exercise_shape:', exercise_keypoint_list.shape)
                        print('user_shape:', user_keypoint_list.shape)
                        
                    elif exercise_keypoint_list.shape[0] / 1.5 < user_keypoint_list.shape[0] and np.array(scoreList).mean() >= score + 5:  # 점수가 이전 점수들에 비해 떨어지는 경향을 보이는 경우
                        scoreList = []  # 점수 기록 초기화
                        share_replay.value = 1  # 재시작
                        print('score down')
                        print('exercise_shape:', exercise_keypoint_list.shape)
                        print('user_shape:', user_keypoint_list.shape)

                    elif score >= 80 and (exercise_keypoint_list.shape[0] / 1.3 < user_keypoint_list.shape[0] < exercise_keypoint_list.shape[0] * 1.3):
                        scoreList = []  # 점수 기록 초기화
                        share_next.value = 1  # 사용자 영상 재 수집 및 다음 영상 요청
                        print('score 80')
                        print('exercise_shape:', exercise_keypoint_list.shape)
                        print('user_shape:', user_keypoint_list.shape)

                    # elif exercise_keypoint_list.shape[0] / 1.5 < user_keypoint_list.shape[0] and np.array(scoreList).mean() > score :  # 점수가 이전 점수들에 비해 떨어지는 경향을 보이는 경우
                    #     scoreList = []  # 점수 기록 초기화
                    #     share_replay.value = 1  # 재시작
                    #     print('score down')
                    #     print('exercise_shape:', exercise_keypoint_list.shape)
                    #     print('user_shape:', user_keypoint_list.shape)

        if user_keypoint_list.shape[0] >= exercise_keypoint_list.shape[0] + 50:  # 사용자가 너무 느리게 따라하는 경우
            scoreList = []
            share_replay.value = 1
            print('list long')
            print('exercise_shape:', exercise_keypoint_list.shape)
            print('user_shape:', user_keypoint_list.shape)

class Video(QThread):
    update_video_frame = pyqtSignal(QtGui.QImage)
    setLabel = pyqtSignal(str, str)
    showDlg = pyqtSignal(str, str, str, str)
    exitDlg = pyqtSignal(str, str)
    kill_thread = pyqtSignal()
    setExStartText = pyqtSignal(int)

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
            self.setLabel.emit(exercise_name, exercise_level)

            pickle_dir = self.pickle_dir + exercise_name + '.pickle'  # Find pickleFile dir

            exercise_keypoint_list = load(lookupfile_path=pickle_dir,
                                          activity_name=exercise_name).array()  # pickle file loaded

            cap = cv2.VideoCapture(file_name)
            while cap.isOpened():
                if score_value.value == 0: # 0점일 때 '운동을 시작해주세요' 띄워야한다~
                    self.setExStartText.emit(1)
                else:
                    self.setExStartText.emit(0)
                
                if exit_value == 1:  # ESC나 종료 버튼 클릭
                    self.exitDlg.emit(str(0), str(0))
                    time.sleep(100)
                    #self.kill_thread.emit()

                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(
                        cv2.CAP_PROP_FRAME_COUNT):  # 영상의 현재 프레임이 영상의 최대 프레임(끝)과 같을때, 즉 영상의 끝일 때
                    if len(exercise_keypoint_list) == 0:  # 영상 list안이 비어있을 때  User Class에서 exercise_keypoint_list를 비워버렸을 때 다음 영상으로 넘어가야한다.
                        last_score = str(score_value.value)
                        score_value.value = 0  # Score를 0으로 setting
                        print('next')
                        print(counter)
                        if counter == len(all_file):
                            self.exitDlg.emit(all_file[counter-1].split('/')[2].split('_')[1], last_score)
                            print("end")
                            time.sleep(100)
                            # self.kill_thread.emit()
                        else:
                            self.showDlg.emit(all_file[counter].split('/')[2].split('_')[1], exercise_level, last_score, all_file[counter-1].split('/')[2].split('_')[1])
                        time.sleep(3)
                        next_video_value.value = 0
                        break

                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video replay

                ret, frame = cap.read()
                # 이미지를 다시 RGB형식으로 칠함 (먼저는 프레임을 잡아줘야한다)
                frame = cv2.flip(frame, 1)  # 양수: 좌우 대칭
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  # 이미지 다시쓰기

                image = cv2.resize(image, (800, 900), interpolation=cv2.INTER_CUBIC)

                time.sleep(1 / FPS)  # Adjust frame speed

                h, w, c = image.shape  # height, weight, channel
                qImg = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
                self.update_video_frame.emit(qImg)  # update video frame

            cap.release()  # free memory


class User(QThread):
    update_user_frame = pyqtSignal(QtGui.QImage)
    setScore = pyqtSignal(str)
    setCamText = pyqtSignal()

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
                    status, frame = cam.read()  # 카메라의 상태 및 프레임 받아옴, 정상 작동일 경우 status = True

                    if status:
                        self.setScore.emit(str(score_value.value))  # score 업데이트
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색상 공간 변환 함수
                        results = pose.process(frame)
                        frame.flags.writeable = True
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                  landmark_drawing_spec=drawing_spec)

                        frame = cv2.resize(frame, dsize=(800, 900), interpolation=cv2.INTER_CUBIC)
                        frame = cv2.flip(frame, 1)  # 양수: 좌우 대칭

                        h, w, c = frame.shape  # height, weight, channel
                        qImg = QImage(frame.data, w, h, w * c, QImage.Format_RGB888)
                        self.update_user_frame.emit(qImg)

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

                            if np.array(user_keypoint_list).shape[0] >= 30 and np.array(exercise_keypoint_list).shape[
                                0] > 0:  # 둘다 길이가 일정 이상일 때
                                if user_queue.qsize() == 0 and exercise_queue.qsize() == 0:
                                    user_queue.put(np.array(user_keypoint_list))  # 프로세스간 공유 자원 큐에 list를 넣는다.
                                    exercise_queue.put(np.array(exercise_keypoint_list))

                        else:  # pose landmark가 보이지 않는다면 지금까지 추출되었던 list 초기화
                            user_keypoint_list = []
                            # print('pose landmark가 보이지 않는다면 지금까지 추출되었던 list 초기화')
                            score_value.value = 0

                    else:
                        print("cannot read frame")
                        self.setCamText.emit()
                        break

                cam.release()  # free memory


# 화면을 띄우는데 사용되는 Class 선언
class Window(QMainWindow, form_class):
    counter = 0
    score_list = []

    def __init__(self):
        self.dlg = DialogWindow()
        self.eDlg = ExitDialogWindow()
        global exercise_queue
        global user_queue
        global score_value
        global next_video_value
        global replay_value

        super().__init__()
        self.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)  # 상단바 제거
        self.show()
        # user cam thread 생성
        self.user_th = User()
        self.user_th.update_user_frame.connect(self.update_user_frame)
        self.user_th.setScore.connect(self.setScore)
        self.user_th.setCamText.connect(self.setCamText)
        self.user_th.start()

        # exercise video thread 생성
        self.ex_th = Video()
        self.ex_th.update_video_frame.connect(self.update_video_frame)
        self.ex_th.setLabel.connect(self.setLabel)
        self.ex_th.showDlg.connect(self.displayDialog)
        self.ex_th.exitDlg.connect(self.ExitDlg)
        self.ex_th.kill_thread.connect(self.kill_thread)
        self.ex_th.setExStartText.connect(self.setExStartText)
        self.ex_th.start()

        # 버튼 클릭 이벤트 함수 연결 - 종료하기
        self.exit_button.clicked.connect(self.exit)
        self.eDlg.exit_button.clicked.connect(self.kill_thread)
        self.p = Process(target=process, args=(exercise_queue, user_queue, score_value, replay_value, next_video_value))
        self.p.start()

    def exit(self):
        global exit_value
        exit_value = 1

    @pyqtSlot(str, str)
    def setLabel(self, name, level):
        self.exercise_name.setText(name)
        self.exercise_level.setText(level)
    
    @pyqtSlot(int)
    def setExStartText(self, isStart):
        if isStart:
            self.label_3.setText("운동을 시작해주세요")
            self.label_3.setStyleSheet("background-color : white;" 
                                       "color:black;"
                                       "font-weight:600;"
                                       "border-radius:20px")
        else:
            self.label_3.setText("운동중입니다")
            self.label_3.setStyleSheet("background-color : #3c4043;" 
                                       "color:white;")
        
    
    @pyqtSlot(str)
    def setScore(self, score):
        self.exercise_score.setText(score)

    @pyqtSlot(QImage)
    def update_user_frame(self, image):
        self.user_cam.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def setCamText(self):
        self.user_cam.setText("카메라가 연결되어 있지 않습니다.")

    @pyqtSlot(QImage)
    def update_video_frame(self, image):
        self.exercise_cam.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def showDlg(self):
        self.dlg = DialogWindow()

    def kill_thread(self):
        self.user_th.quit()  # thread 종료
        self.ex_th.quit()
        self.p.terminate()
        self.eDlg.eDlgClose()
        self.close()
        exit(0)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:  # ESC 누르면 종료
            global exit_value
            exit_value = 1

        if e.key() == Qt.Key_N:
            next_video_value.value = 1

    @pyqtSlot(str, str, str, str)
    def displayDialog(self, ex_name, ex_level, ex_score, cur_ex_name):
        self.counter += 1
        self.score_list.append(float(ex_score))
        self.eDlg.addText(cur_ex_name, ex_score)
        self.dlg.setDialogText(ex_name, ex_level, ex_score)
        self.dlg.show()
        self.dlg.startTimer()

    @pyqtSlot(str, str)
    def ExitDlg(self, last_exercise, last_score):
        if last_exercise != "0" or last_score != "0":  # 마지막 운동 영상일 때
            self.score_list.append(float(last_score))
            self.counter += 1
            self.eDlg.addText(last_exercise, last_score)
        if self.counter == 0:
            self.eDlg.addText("진행한 운동 없음", str(0))
        else:
            self.eDlg.setResultScore(round((sum(self.score_list)/self.counter), 2))
        self.eDlg.show()

    # MOUSE Click drag EVENT function
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # Get the position of the mouse relative to the window
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # Change mouse icon

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # Change window position
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


class startWindow(QWidget, form_start):
    def __init__(self):
        super(startWindow, self).__init__()
        self.main = None
        self.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        pixmap = QPixmap('ui/logo.png')
        self.logo.setPixmap(pixmap)

        self.start_button.clicked.connect(self.btn_start_to_main)

    def btn_start_to_main(self):
        self.hide()  # 시작화면 숨김
        time.sleep(1)
        self.main = Window()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:  # ESC 누르면 종료
            exit(0)


class DialogWindow(QDialog):

    def __init__(self):  # 부모 window 설정
        super(DialogWindow, self).__init__()
        self.sec = 3
        dialog_ui = 'ui/next_video_dialog.ui'
        uic.loadUi(dialog_ui, self)

        self.myTimer = QTimer(self)
        self.myTimer.timeout.connect(self.timerTimeout)
        self.second.setText(str(self.sec))
        self.center()

    # 통과한 레벨 출력
    def setLevelText(self, level):
        self.level.setText(level)

    # 평균 정확도 출력
    def setAverageScore(self, score):
        self.avg_score.setText(score)

    # 다음 운동 이름 출력
    def setNextExName(self, name):
        self.next_ex.setText(name)

    def startTimer(self):
        self.myTimer.setInterval(1000)
        self.myTimer.start()  # 1초 마다 timerTimeout 함수 실행

    def timerTimeout(self):
        self.sec -= 1
        self.second.setText(str(self.sec))  # 몇 초 남았는지 set
        if self.sec == 0:
            self.myTimer.stop()
            self.sec = 3
            self.second.setText(str(self.sec))
            self.close()

    def setDialogText(self, ex_name, level, score):
        self.setAverageScore(f":  {score}")
        self.setLevelText(f":  {level}")
        self.setNextExName(f":  {ex_name}")


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class ExitDialogWindow(QDialog):
    scoreText = ""
    exNameText =""
    def __init__(self):
        super(ExitDialogWindow, self).__init__()
        dialog_ui = 'ui/exit_dialog.ui'
        uic.loadUi(dialog_ui, self)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def addText(self, ex_name, ex_score):
        self.exNameText += ex_name + '\n\n'
        self.scoreText += ex_score + '\n\n'
        self.setExitText()

    def setExitText(self):
        self.ex_name.setText(self.exNameText)
        self.ex_score.setText(self.scoreText)

    def setResultScore(self, result):
        self.result_score.setText(str(result))

    def eDlgClose(self):
        self.close()

if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # Window Class의 인스턴스 생성
    # myWindow = Window()
    startWindow = startWindow()
    startWindow.show()
    # myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    sys.exit(app.exec_())
