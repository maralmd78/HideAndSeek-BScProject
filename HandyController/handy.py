import typing
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QByteArray, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QIntValidator
from PyQt6.QtNetwork import QTcpSocket, QHostAddress, QAbstractSocket
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
        self.ip = self.lineEdit_IP.text()
        self.port = int(self.lineEdit_Port.text())

        self.socket = QTcpSocket()
        self.networkConnection()

        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.handleTimer)
        self.timer.start()


    

    @pyqtSlot()
    def IpConfig(self):
        self.ip = self.lineEdit_IP.text()
        self.networkConnection()
    
    @pyqtSlot()
    def PortConfig(self):
        self.port = int(self.lineEdit_Port.text())
        self.networkConnection()
    
    def networkConnection(self):
        self.socket.disconnectFromHost()
        self.socket.connectToHost(self.ip, self.port)
        ## waiting 5 sec
        if not self.socket.waitForConnected(5000):
            print("cannot connect to server", self.socket.errorString())
        else:
            print("successfully connected to", self.ip, self.port)
            # data = {"S1":120,"S2":80,"S3":70,"D1":1,"D2":0,"D3":1}
            # self.socket.write(QByteArray(bytes(json.dumps(data, separators=(',',':')), 'utf-8')))
            # self.socket.waitForBytesWritten(1000)
            print("here")
    
    @pyqtSlot()
    def handleTimer(self):
        if self.socket.state() == QAbstractSocket.SocketState.ConnectedState:
            data = {"S1":120,"S2":80,"S3":70,"D1":1,"D2":0,"D3":1}
            self.socket.write(QByteArray(bytes(json.dumps(data, separators=(',',':')), 'utf-8')))            
        

    





app = QApplication(sys.argv)
mainwindow = MainWindow()
mainwindow.show()
app.exec()