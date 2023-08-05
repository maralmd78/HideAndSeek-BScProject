from OpenGL import GL as gl
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from simulator.particlesim import ParticleSim
from PyQt6.QtGui import QPainter, QPen, QTransform
from PyQt6.QtCore import Qt, QTimer, QPointF, QLineF
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import sys


class GUI(QOpenGLWidget):
    def __init__(self, ps: ParticleSim):
        super().__init__()
        self.setWindowTitle("Particle Simulator")
        self.resize(800, 800)
        self.ps = ps
        self.painter = QPainter()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handle_timer)
        self.timer.start(10)

    def handle_timer(self):
        self.update()

    def initializeGL(self):
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.painter.begin(self)

        transform = QTransform()
        transform.translate(self.width()/2, self.height()/2)  # move origin to center of widget
        transform.scale(self.width()/200.0, -self.height()/200.0)  # map x and y to range -100 to 100
        self.painter.setTransform(transform)

        for id, row in enumerate(self.ps.robots_pos):
            self.painter.setPen(QPen(self.ps.robots_color[id], 0.1))
            self.painter.setBrush(self.ps.robots_color[id])
            self.painter.drawEllipse(QPointF(*row), self.ps.ROBOT_RADIUS, self.ps.ROBOT_RADIUS)

        self.painter.setPen(QPen(Qt.GlobalColor.lightGray, 0.5))
        self.painter.drawLines([QLineF(*line.p1, *line.p2) for line in self.ps.lines])

        self.painter.setPen(QPen(Qt.GlobalColor.black, 1))
        self.painter.drawLines([QLineF(*wall.p1, *wall.p2) for wall in self.ps.walls])

        
        self.painter.end()




if __name__ == '__main__':
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
    app = QApplication(sys.argv)

    ps = ParticleSim()
    gui = GUI(ps=ps)
    gui.show()

    sys.exit(app.exec())
