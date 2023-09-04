from simulator.particlesim import ParticleSim
from execute.control import PID, GotoPoint
import threading
import time
from sympy import Point, Segment
from PyQt6.QtCore import Qt
import numpy as np
import math
from .grid import Grid
from .maze import MazeSim
import pickle

class Execute(threading.Thread):
    def __init__(self, ps: ParticleSim):
        super().__init__()
        self.ps = ps
        
        self.seekerRobot = ps.add_robot(pos=Point(0, 0), color=Qt.GlobalColor.blue)
 
    def run(self):
        counter = 0
        controller = GotoPoint(ps=self.ps, robotID=self.seekerRobot, pid_coef=(20, 0, 0),threshold=0.05)
        self.ps.add_line(Segment((-30, 0), (30, 0)))
        self.ps.add_line(Segment((0, -30), (0, 30)))
        setpoint = Point(20, 20)
        positions = []
        errors = []
        times = []
        while True:
            command = controller.update(setpoint)
            self.ps.robot_command(self.seekerRobot, command)
            position_feedback = self.ps.robot_feedback(self.seekerRobot)[0]
            positions.append((position_feedback.x, position_feedback.y))
            square_error = np.linalg.norm(np.array(setpoint, dtype=float) - np.array(position_feedback, dtype=float), 2)
            errors.append(square_error)
            times.append(counter*self.ps.delta_t)
            counter += 1
            if len(positions) == 300:
                with open('error_pos(20, 20).pkl', 'wb') as f:  
                    pickle.dump([errors, positions, times, setpoint], f)
                print("finish!")
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
            time.sleep(0.01)




