from simulator.particlesim import ParticleSim
from execute.execute import Execute
from simulator.gui import GUI
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import sys


QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)
app = QApplication(sys.argv)

ps = ParticleSim(delta_t=0.01)
gui = GUI(ps=ps)
gui.show()

exe = Execute(ps=ps)
exe.start()

sys.exit(app.exec())

