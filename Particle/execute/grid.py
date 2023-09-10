import numpy as np
from sympy import Point, Segment
from simulator.particlesim import ParticleSim
from PyQt6.QtCore import Qt, QRectF
from execute.control import GotoPoint
from execute.maze import MazeSim, Qlearning
import pickle

class Grid:
    def __init__(self, ps:ParticleSim, maze:MazeSim) -> None:
        self.ps = ps
        self.maze = maze

        # with open('Qtable_train_seeker&hider_final-50000.pkl', "rb") as f:  # Python 3: open(..., 'rb')
        #     self.Qtable_seeker, self.Qtable_hider = pickle.load(f)
        ########## for loading the q table
        # with open('Qtable_train_only_hider-50000.pkl', "rb") as f:  # Python 3: open(..., 'rb')
        #     Qtable_seeker_init, Qtable_hider_init = pickle.load(f)
        #     # self.Qtable_seeker, self.Qtable_hider = pickle.load(f)
        ############ for first train with initial values 0 for  both q tables
        # Qtable_seeker_init = np.zeros((len(maze.state_space), len(maze.action_space)), dtype=float)
        # Qtable_hider_init = np.zeros((len(maze.state_space), len(maze.action_space)), dtype=float)
        # self.Qtable_seeker, self.Qtable_hider, self.rewards_seeker, self.rewards_hider = Qlearning(maze, Qtable_seeker_init, Qtable_hider_init, 300000, 1000, gamma=0.99, alpha=0.2)
        # with open('Qtable_train_seekerOnly13_(maze, 0, 0, 300000, 1000, 0.99, 0.2).pkl', 'wb') as f:  
        #     pickle.dump([self.Qtable_seeker, self.Qtable_hider, self.rewards_seeker, self.rewards_hider], f)
        
        with open('Qtable_train_seekerOnly13_(maze, 0, 0, 300000, 1000, 0.99, 0.2).pkl', "rb") as f:  # Python 3: open(..., 'rb')
            self.Qtable_seeker, self.Qtable_hider, self.rewards_seeker, self.rewards_hider = pickle.load(f)

        # self.policy_seeker = [maze.action_space[np.argmax(row)] if np.all(row != 0) else 'none' for row in self.Qtable_seeker]
        # self.policy_hider = [maze.action_space[np.argmax(row)] if np.all(row != 0) else 'none' for row in self.Qtable_hider]
        
        # with open('Qtable_train_hider50000.pkl', "rb") as f:  # Python 3: open(..., 'rb')
            # self.Qtable_seeker, self.Qtable_hider = pickle.load(f)
        
        self.policy_seeker = [maze.action_space[np.argmax(row)] if np.all(row != 0) else 'none' for row in self.Qtable_seeker]
        self.policy_hider = [maze.action_space[np.argmax(row)] if np.all(row != 0) else 'none' for row in self.Qtable_hider]
        
        self.gridArr = self.maze.maze ## 7*7 array
        self.cellHeight = 200/self.gridArr.shape[0] ## gui:-100 to 100 -->200
        self.cellWidth = 200/self.gridArr.shape[1]

        self.seekerID = None
        self.hiderID = None
        self.seekerController = None
        self.hiderController = None
        # add walls
        for i, row in enumerate(self.gridArr):
            for j, cell in enumerate(row):
                if cell==-1:
                    middlePoint = self.indexToPos(i, j)
                    #up
                    self.ps.add_wall(Segment((middlePoint.x-self.cellWidth/2, middlePoint.y+self.cellHeight/2), (middlePoint.x+self.cellWidth/2, middlePoint.y+self.cellHeight/2)))
                    #left
                    self.ps.add_wall(Segment((middlePoint.x-self.cellWidth/2, middlePoint.y+self.cellHeight/2), (middlePoint.x-self.cellWidth/2, middlePoint.y-self.cellHeight/2)))
                    #right
                    self.ps.add_wall(Segment((middlePoint.x+self.cellWidth/2, middlePoint.y+self.cellHeight/2), (middlePoint.x+self.cellWidth/2, middlePoint.y-self.cellHeight/2)))
                    #down
                    self.ps.add_wall(Segment((middlePoint.x-self.cellWidth/2, middlePoint.y-self.cellHeight/2), (middlePoint.x+self.cellWidth/2, middlePoint.y-self.cellHeight/2)))
                    #rect
                    self.ps.add_rect(QRectF(middlePoint.x-self.cellWidth/2, middlePoint.y+self.cellHeight/2, self.cellWidth, -self.cellHeight))
        
        # add lines
        # for i in range(self.gridArr.shape[0]):
        #     self.ps.add_line(Segment((-100, 100-(i*self.cellHeight)), (100, 100-(i*self.cellHeight))))
        # for i in range(self.gridArr.shape[1]):
        #     self.ps.add_line(Segment((-100+(i*self.cellWidth), -100), (-100+(i*self.cellWidth), 100)))
        


    
    def indexToPos(self, row, column):
        x = -100 + column*self.cellWidth + self.cellWidth/2
        y = 100 - (row*self.cellHeight) - self.cellHeight/2
        return Point(x, y)
    
    def positionToIndex(self, pos:Point):
        x = 100 + pos.x
        y = 100 - pos.y
        return (y//self.cellHeight, x//self.cellWidth)
    
    def step(self):
        i_s, j_s = self.positionToIndex(self.ps.robot_feedback(self.seekerID)[0])
        i_h, j_h = self.positionToIndex(self.ps.robot_feedback(self.hiderID)[0])
        state = self.maze.index_2_state((i_s, j_s), (i_h, j_h))
        action_seeker = self.policy_seeker[state]
        action_hider = self.policy_hider[state]
        print(action_seeker, action_hider)
        
        # seeker
        if action_seeker == "up":
            setpoint_seeker = self.indexToPos(i_s-1, j_s)
        elif action_seeker == "down":
            setpoint_seeker = self.indexToPos(i_s+1, j_s)
        elif action_seeker == "left":
            setpoint_seeker = self.indexToPos(i_s, j_s-1)
        elif action_seeker == "right":
            setpoint_seeker = self.indexToPos(i_s, j_s+1)
        else:
            setpoint_seeker = self.indexToPos(i_s, j_s)
        
        # hider
        if action_hider == "up":
            setpoint_hider = self.indexToPos(i_h-1, j_h)
        elif action_hider == "down":
            setpoint_hider = self.indexToPos(i_h+1, j_h)
        elif action_hider == "left":
            setpoint_hider = self.indexToPos(i_h, j_h-1)
        elif action_hider == "right":
            setpoint_hider = self.indexToPos(i_h, j_h+1)
        else:
            setpoint_hider = self.indexToPos(i_h, j_h)
    
        if (i_s, j_s) == (i_h, j_h):
            command_seeker = command_hider = Point(0, 0)
        else:
            command_seeker = self.seekerController.update((setpoint_seeker.x, setpoint_seeker.y))
            command_hider = self.hiderController.update((setpoint_hider.x, setpoint_hider.y))
        self.ps.robot_command(self.seekerID, command_seeker)
        self.ps.robot_command(self.hiderID, command_hider)



        
    
    def reset(self, seekerPos:Point, hiderPos:Point):
        self.ps.removeRobots()
        self.seekerID = self.ps.add_robot(seekerPos, Qt.GlobalColor.red)
        self.hiderID = self.ps.add_robot(hiderPos, Qt.GlobalColor.blue)
        self.seekerController = GotoPoint(ps=self.ps, robotID=self.seekerID, pid_coef=(20, 0, 0),threshold=0.05)
        self.hiderController = GotoPoint(ps=self.ps, robotID=self.hiderID, pid_coef=(20, 0, 0),threshold=0.05)
    
    

    


if __name__ == "__main__":
    with open('policy.pkl', "rb") as f:  # Python 3: open(..., 'rb')
        seeker, hider = pickle.load(f)


