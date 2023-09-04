import typing
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QByteArray, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QIntValidator
from PyQt6.QtNetwork import QTcpSocket, QHostAddress, QAbstractSocket, QUdpSocket
from PyQt6 import QtCore, uic
import sys
import json 
import numpy as np



class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi("mainwindow.ui", self)
        self.lineEdit_Port.setValidator(QIntValidator(10, 9999))
        self.lineEdit_IP.editingFinished.connect(self.IpConfig)
        self.lineEdit_Port.editingFinished.connect(self.PortConfig)

        self.lineEdit_speed1.setValidator(QIntValidator(0, 9999))
        self.lineEdit_speed2.setValidator(QIntValidator(0, 9999))
        self.lineEdit_speed3.setValidator(QIntValidator(0, 9999))
        self.lineEdit_speed1.editingFinished.connect(self.speed1Config)
        self.lineEdit_speed2.editingFinished.connect(self.speed2Config)
        self.lineEdit_speed3.editingFinished.connect(self.speed3Config)

        self.lineEdit_direction1.editingFinished.connect(self.direction1Config)
        self.lineEdit_direction2.editingFinished.connect(self.direction2Config)
        self.lineEdit_direction3.editingFinished.connect(self.direction3Config)

        self.pushButton_mode.pressed.connect(self.modePressed)

        self.setFocus()


        
        self.ip = self.lineEdit_IP.text()
        self.port = int(self.lineEdit_Port.text())
        self.speed1 = int(self.lineEdit_speed1.text())
        self.speed2 = int(self.lineEdit_speed2.text())
        self.speed3 = int(self.lineEdit_speed3.text())
        self.direction1 = int(self.lineEdit_direction1.text())
        self.direction2 = int(self.lineEdit_direction2.text())
        self.direction3 = int(self.lineEdit_direction3.text())
        self.q_dot = [0, 0, 0] # [w_bz, v_bx, v_by]
        self.r = 2.5/100 #TODO
        self.d = 7/100 #TODO
        self.H_matrix = np.array([[-self.d, 1, 0], [-self.d, -1/2, -np.sin(np.pi/3)], [-self.d, -1/2, np.sin(np.pi/3)]])
        self.stepperRev = 200

        self.socket = QUdpSocket()
        self.networkConnection()

        
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.handleTimer)
        self.timer.start()

        
    def sign(self, num):
        return (np.sign(num) + 1)//2
    
    def inverseVelocityEq(self):
        u = (1/self.r)*(self.H_matrix)@(np.array(self.q_dot))
        u = (u/(2*np.pi))*self.stepperRev
        data = {"S1":np.abs(u[0]),"S2":np.abs(u[1]),"S3":np.abs(u[2]),"D1":self.sign(u[0]),"D2":self.sign(u[1]),"D3":self.sign(u[2])}
        return data

    @pyqtSlot()
    def IpConfig(self):
        self.ip = self.lineEdit_IP.text()
        self.networkConnection()
    
    @pyqtSlot()
    def PortConfig(self):
        self.port = int(self.lineEdit_Port.text())
        self.networkConnection()
    
    @pyqtSlot()
    def speed1Config(self):
        self.speed1 = int(self.lineEdit_speed1.text())
    
    @pyqtSlot()
    def speed2Config(self):
        self.speed2 = int(self.lineEdit_speed2.text())
    
    @pyqtSlot()
    def speed3Config(self):
        self.speed3 = int(self.lineEdit_speed3.text())

    
    @pyqtSlot()
    def direction1Config(self):
        self.direction1 = int(self.lineEdit_direction1.text())
    
    @pyqtSlot()
    def direction2Config(self):
        self.direction2 = int(self.lineEdit_direction2.text())
    
    @pyqtSlot()
    def direction3Config(self):
        self.direction3 = int(self.lineEdit_direction3.text())
    
    @pyqtSlot()
    def modePressed(self):
        if self.pushButton_mode.text() == "keyboard":
            self.pushButton_mode.setText("S/D") #direct wheel speed/direction
        elif self.pushButton_mode.text() == "S/D":
            self.pushButton_mode.setText("keyboard")
        
    
    @pyqtSlot(QKeyEvent)
    def keyPressEvent(self, event:QKeyEvent):
        if event.key() == Qt.Key.Key_Up:
            self.q_dot = [0, 0, 0.2] #TODO
        elif event.key() == Qt.Key.Key_Right:
            self.q_dot = [0, 0.2, 0] #TODO
        elif event.key() == Qt.Key.Key_Down:
            self.q_dot = [0, 0, -0.2] #TODO
        elif event.key() == Qt.Key.Key_Left:
            self.q_dot = [0, -0.2, 0] #TODO
        elif event.key() == Qt.Key.Key_Q:
            self.q_dot = [-np.pi/6, 0, 0] #TODO
        elif event.key() == Qt.Key.Key_W:
            self.q_dot = [np.pi/6, 0, 0] #TODO
        elif event.key() == Qt.Key.Key_Space:
            self.q_dot = [0, 0, 0] #TODO
        if event.key() == Qt.Key.Key_Escape:
            self.setFocus()
       
    def networkConnection(self):
        # self.socket.disconnectFromHost()
        # self.socket.connectToHost(self.ip, self.port)
        # ## waiting 5 sec
        # if not self.socket.waitForConnected(5000):
        #     print("cannot connect to server", self.socket.errorString())
        # else:
        #     print("successfully connected to", self.ip, self.port)
        self.socket.disconnectFromHost()
        if not self.socket.bind(QHostAddress(self.ip), self.port):
            print("ERROR in binding the socket")            
    
    def sendPacket(self):
        if self.pushButton_mode.text() == "S/D":
            data = {"S1":self.speed1,"S2":self.speed2,"S3":self.speed3,"D1":self.direction1,"D2":self.direction2,"D3":self.direction3}
        elif self.pushButton_mode.text() == "keyboard":
            data = self.inverseVelocityEq()
        # data = {"S1":100,"S2":300,"S3":500,"D1":0,"D2":0,"D3":0}
        # print(data)
        self.socket.writeDatagram(QByteArray(bytes(json.dumps(data), 'utf-8')), QHostAddress(self.ip), self.port)
    
    @pyqtSlot()
    def handleTimer(self):
        self.sendPacket()
    
        

    





app = QApplication(sys.argv)
mainwindow = MainWindow()
mainwindow.show()
app.exec()