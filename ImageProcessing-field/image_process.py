from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QByteArray, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QIntValidator
from PyQt6.QtNetwork import QUdpSocket, QHostAddress, QAbstractSocket
from PyQt6 import uic
import cv2
import sys
import json 
import numpy as np
import argparse

class thread_cv(QThread):
    new_frame_signal = pyqtSignal(QImage)
    
    def __init__(self):
        super().__init__()
        self.position = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.sendPosition)
        self.timer.setInterval(250)
        self.timer.start()

        self.rectangle = [[218, 84], [482, 352]]
        self.rotation = 4
        self.calibration_process = False
        self.field_height = 120
        self.field_width = 120
        self.h_min = 0
        self.h_max = 0
        self.s_min = 0
        self.s_max = 0
        self.v_min = 0
        self.v_max = 0
        self.hsv_mode = "NORMAL"
        self.socket = QUdpSocket()
        if not self.socket.bind(QHostAddress.SpecialAddress.LocalHost, 1234, QAbstractSocket.BindFlag.ShareAddress | QAbstractSocket.BindFlag.ReuseAddressHint):
            print("ERROR in binding the socket")
    
    def sendPosition(self):
        if self.position is not None:
            self.socket.writeDatagram(QByteArray(bytes(json.dumps(self.position), 'utf-8')), QHostAddress.SpecialAddress.LocalHost, 1234)

    def run(self):
        # cap = cv2.VideoCapture('output.avi')
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = self.image_processing(frame)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_BGR888)
                self.new_frame_signal.emit(image)
                cv2.waitKey(1)
    
    def image_processing(self, frame):
        frame = self.rotate_image(frame, self.rotation)
        if self.calibration_process:
            frame = cv2.rectangle(frame, tuple(self.rectangle[0]), tuple(self.rectangle[1]), (0, 0, 255), 4)
            # print(self.rectangle, self.rotation)
        else:
            mask = np.zeros(frame.shape[:2], np.uint8)
            mask[self.rectangle[0][1]:self.rectangle[1][1], self.rectangle[0][0]:self.rectangle[1][0]] = 255
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_HSV = cv2.inRange(frame_HSV, (self.h_min, self.s_min, self.v_min), (self.h_max, self.s_max, self.v_max))
            
            positions = np.array(np.where(mask_HSV == 255))
            
            if self.hsv_mode == "BLOB":
                frame = cv2.bitwise_and(frame, frame, mask=mask_HSV)
                
            if positions.shape[1] > 0:
                center = np.round(np.mean(positions, axis=1, dtype=int))
                cv2.circle(frame, tuple(center[::-1]), 2, (0, 0, 255), 2)
                
                rect_width_px = self.rectangle[1][0] - self.rectangle[0][0]
                rect_height_px = self.rectangle[1][1] - self.rectangle[0][1]
                
                px_cm_width = self.field_width / rect_width_px
                px_cm_height = self.field_height / rect_height_px
                
                center_wrt_rect = center - self.rectangle[0][::-1]
                position = center_wrt_rect * np.array([px_cm_height, px_cm_width]) # [height(y), width(x)]
                position = position[::-1] # [x, y]
                self.position = {'x': np.round(position[0], 2), 'y': np.round(position[1], 2)}
                 
        
        return frame
    
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @pyqtSlot()
    def border_pressed(self):
        self.calibration_process = True
    
    @pyqtSlot()
    def fix_pressed(self):
        self.calibration_process = False
    
    @pyqtSlot(str)
    def height_changed(self, txt:str):
        try:
            self.field_height = int(txt)
        except:
            self.field_height = 0
            print("INVALID HEIGHT DIMENSION")
         
    @pyqtSlot(str)
    def width_changed(self, txt:str):
        try:
            self.field_width = int(txt)
        except:
            self.field_width = 0
            print("INVALID WIDTH DIMENSION")
    
    @pyqtSlot(str)
    def h_min_changed(self, txt:str):
        try:
            self.h_min = int(txt)
        except:
            self.h_min = 0
            print("INVALID HUE MIN DIMENSION")
    
    @pyqtSlot(str)
    def h_max_changed(self, txt:str):
        try:
            self.h_max = int(txt)
        except:
            self.h_max = 0
            print("INVALID HUE MAX DIMENSION")
    
    @pyqtSlot(str)
    def s_min_changed(self, txt:str):
        try:
            self.s_min = int(txt)
        except:
            self.s_min = 0
            print("INVALID SATURATION MIN DIMENSION")
    
    @pyqtSlot(str)
    def s_max_changed(self, txt:str):
        try:
            self.s_max = int(txt)
        except:
            self.s_max = 0
            print("INVALID SATURATION MAX DIMENSION")
    
    @pyqtSlot(str)
    def v_min_changed(self, txt:str):
        try:
            self.v_min = int(txt)
        except:
            self.v_min = 0
            print("INVALID VALUE MIN DIMENSION")
    
    @pyqtSlot(str)
    def v_max_changed(self, txt:str):
        try:
            self.v_max = int(txt)
        except:
            self.v_max = 0
            print("INVALID VALUE MAX DIMENSION")
    
    @pyqtSlot()
    def normal_pressed(self):
        self.hsv_mode = "NORMAL"
    
    @pyqtSlot()
    def blob_pressed(self):
        self.hsv_mode = "BLOB"
    
    @pyqtSlot(QKeyEvent)
    def keyPressEvent(self, event:QKeyEvent):
        if self.calibration_process:
            if event.key() == Qt.Key.Key_Up:
                self.rectangle[0][1] -= 2
                self.rectangle[1][1] -= 2
            elif event.key() == Qt.Key.Key_Right:
                self.rectangle[0][0] += 2
                self.rectangle[1][0] += 2
            elif event.key() == Qt.Key.Key_Down:
                self.rectangle[0][1] += 2
                self.rectangle[1][1] += 2
            elif event.key() == Qt.Key.Key_Left:
                self.rectangle[0][0] -= 2
                self.rectangle[1][0] -= 2
                
            elif event.key() == Qt.Key.Key_W:
                self.rectangle[0][1] -= 2
                self.rectangle[1][1] += 2
            elif event.key() == Qt.Key.Key_D:
                self.rectangle[0][0] -= 2
                self.rectangle[1][0] += 2
            elif event.key() == Qt.Key.Key_S:
                self.rectangle[0][1] += 2
                self.rectangle[1][1] -= 2
            elif event.key() == Qt.Key.Key_A:
                self.rectangle[0][0] += 2
                self.rectangle[1][0] -= 2
            
            elif event.key() == Qt.Key.Key_N:
                self.rotation += 2
            elif event.key() == Qt.Key.Key_M:
                self.rotation -= 2
        


class MainWindow(QMainWindow):
    key_pressed_signal = pyqtSignal(QKeyEvent)
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("mainwindow.ui", self)
        self.setWindowTitle("Pursuit and Evade")
        
        self.thread = thread_cv()
        self.key_pressed_signal.connect(self.thread.keyPressEvent)
        self.thread.new_frame_signal.connect(self.setImage)
        self.button_border.pressed.connect(self.thread.border_pressed)
        self.button_fix.pressed.connect(self.thread.fix_pressed)
        
        self.lineedit_height.setValidator(QIntValidator(1, 500))
        self.lineedit_width.setValidator(QIntValidator(1, 500))
        self.lineedit_height.textChanged.connect(self.thread.height_changed)
        self.lineedit_width.textChanged.connect(self.thread.width_changed)
        self.lineedit_height.textChanged.emit(self.lineedit_height.text())
        self.lineedit_width.textChanged.emit(self.lineedit_width.text())
        
        self.lineedit_h_min.setValidator(QIntValidator(1, 255))
        self.lineedit_h_max.setValidator(QIntValidator(1, 255))
        self.lineedit_s_min.setValidator(QIntValidator(1, 255))
        self.lineedit_s_max.setValidator(QIntValidator(1, 255))
        self.lineedit_v_min.setValidator(QIntValidator(1, 255))
        self.lineedit_v_max.setValidator(QIntValidator(1, 255))
        self.lineedit_h_min.textChanged.connect(self.thread.h_min_changed)
        self.lineedit_h_max.textChanged.connect(self.thread.h_max_changed)
        self.lineedit_s_min.textChanged.connect(self.thread.s_min_changed)
        self.lineedit_s_max.textChanged.connect(self.thread.s_max_changed)
        self.lineedit_v_min.textChanged.connect(self.thread.v_min_changed)
        self.lineedit_v_max.textChanged.connect(self.thread.v_max_changed)
        self.lineedit_h_min.textChanged.emit(self.lineedit_h_min.text())
        self.lineedit_h_max.textChanged.emit(self.lineedit_h_max.text())
        self.lineedit_s_min.textChanged.emit(self.lineedit_s_min.text())
        self.lineedit_s_max.textChanged.emit(self.lineedit_s_max.text())
        self.lineedit_v_min.textChanged.emit(self.lineedit_v_min.text())
        self.lineedit_v_max.textChanged.emit(self.lineedit_v_max.text())
        
        self.button_normal.pressed.connect(self.thread.normal_pressed)
        self.button_blob.pressed.connect(self.thread.blob_pressed)
        self.thread.start()
        self.setFocus()
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label_image.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot(QKeyEvent)
    def keyPressEvent(self, event:QKeyEvent):
        self.key_pressed_signal.emit(event)
        if event.key() == Qt.Key.Key_Escape:
            self.setFocus()
        


parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="headless mode (without GUI)", action="store_true")
args = parser.parse_args()

app = QApplication(sys.argv)
mainwindow = MainWindow()
if not args.headless:
    mainwindow.show()
app.exec()