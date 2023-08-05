import numpy as np
from sympy import Point
import math
from simulator.particlesim import ParticleSim


class PID:
    def __init__(self, p, i, d) -> None:
        self.p = p
        self.i = i
        self.d = d
        self.errorSum = np.zeros(30)
        self.errorCounter = 0
    
    def update(self, error):
        if self.errorCounter == len(self.errorSum): self.errorCounter = 0
        self.errorSum[self.errorCounter] = error
        self.errorCounter += 1
        if error < 1: pTerm = self.p*error*5
        else: pTerm = self.p*error
        iTerm = self.i*(np.sum(self.errorSum))
        dTerm = self.d*(error - self.errorSum[self.errorCounter-1])
        return pTerm + iTerm + dTerm

    def reset(self):
        self.errorSum = np.zeros(30)
        self.errorCounter = 0

class GotoPoint:
    def __init__(self, ps:ParticleSim, robotID:int, pid_coef, threshold) -> None:
        self.ps = ps
        self.id = robotID
        self.controller = PID(*pid_coef)
        self.threshold = threshold
        self.setPoint = (5000, 5000)
    
    def update(self, setpoint):
        if np.linalg.norm(np.array(setpoint, dtype=float) - np.array(self.setPoint, dtype=float), 2) > self.threshold: 
            self.controller.reset()
        self.setPoint = setpoint
        error = np.array(setpoint) - np.array(self.ps.robot_feedback(self.id)[0])
        error = error.astype(float)
        distance = np.linalg.norm(error, 2)

        if distance <= self.threshold:
            return Point(0, 0)

        u = self.controller.update(distance) 
        return Point(u*(error/distance))
        
        
class GotoPoint2:
    def __init__(self, robotID, x_axis_pid_coef, y_axis_pid_coef, threshold) -> None:
        self.id = robotID
        self.xController = PID(*x_axis_pid_coef)
        self.yController = PID(*y_axis_pid_coef)
        self.threshold = threshold
        self.setPoint = (5000, 5000)
    
    def update(self, setpoint, xFeedback, yFeedback):
        if np.linalg.norm(np.array(setpoint) - np.array(self.setPoint), 2) > self.threshold: 
            self.xController.reset()
            self.yController.reset()
        
        # Xerror = setpoint[0] - self.ps.robot_feedback(0)[0][0]
        Xerror = setpoint[0] - xFeedback 
        Yerror = setpoint[1] - yFeedback
        if np.abs(Xerror) < self.threshold and np.abs(Yerror) < self.threshold:
            print("stop")
            return Point(0, 0)
        
        return Point(self.xController.update(Xerror), self.yController.update(Yerror))



