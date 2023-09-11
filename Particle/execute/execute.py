from simulator.particlesim import ParticleSim
from execute.control import PID, GotoPoint
import threading
import time
from sympy import Point, Segment
from PyQt6.QtCore import Qt
import numpy as np
import math
from .grid import Grid
# from .mazeSim import MazeSim
from .maze import MazeSim

class Execute(threading.Thread):
    def __init__(self, ps: ParticleSim):
        super().__init__()
        self.ps = ps
        self.maze = MazeSim()
        self.grid = Grid(self.ps, self.maze)
        # self.seekerRobot = ps.add_robot(pos=Point(-3*20, 0.5*20), color=Qt.GlobalColor.blue)
 
    def run(self):
        # counter = 0
        # controller = GotoPoint(ps=self.ps, robotID=0, pid_coef=(20, 0, 0),threshold=0.05)
        
        self.grid.reset(seekerPos=self.grid.indexToPos(1, 5), hiderPos=self.grid.indexToPos(5, 1))
        
        while True:
            self.grid.step()

            # if counter<50:
            #     command = controller.update((1*20, 0*20))
            #     # self.ps.robot_command(controller.id, command)
            # elif counter<100:
            #     # self.ps.removeRobots()
            #     command = controller.update((2*20, 2*20))
            # # elif counter<140:
            # #     command = controller.update((-2*20, -2*20))
            # # else:
            # # command = controller.update((-4*20, 2*20))
            # self.ps.robot_command(controller.id, command)
            
         
            # counter += 1
            self.ps.step()
            time.sleep(0.008)




