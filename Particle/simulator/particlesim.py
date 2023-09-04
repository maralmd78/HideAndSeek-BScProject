from sympy import Point, Segment
from sympy.geometry import intersection
import numpy as np
from PyQt6.QtCore import Qt

def intersect(seg1: Segment, seg2: Segment):
    if type(seg1) == type(Point(0, 0)) or type(seg2) == type(Point(0, 0)):
        return None
    
    x1, y1 = seg1.p1.x, seg1.p1.y
    x2,y2 = seg1.p2.x, seg1.p2.y
    x3,y3 = seg2.p1.x, seg2.p1.y
    x4,y4 = seg2.p2.x, seg2.p2.y
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return Point(x,y)
    


class ParticleSim():
    ROBOT_RADIUS = 3.0
    def __init__(self, delta_t=0.01) -> None:
        # 'delta_t' is the step time of each step of the simulator
        self.delta_t = delta_t
        # nx2 matrix containing x and y positions of each robot (n is the number of robots)
        self.robots_pos = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        # nx2 matrix containing x and y velocities of each robot (n is the number of robots)
        self.robots_vel = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        # nx2 matrix containing x and y command velocities of each robot (n is the number of robots)
        self.robots_command_vel = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        # colors of each robot
        self.robots_color = []
        # store walls in a list - add boundy walls first - other walls can be add using 'add_wall' function
        self.walls = []
        self.walls.append(Segment(Point(-100, 100), Point(100, 100))) #up
        self.walls.append(Segment(Point(100, 100), Point(100, -100))) #right
        self.walls.append(Segment(Point(-100, -100), Point(100, -100))) #down
        self.walls.append(Segment(Point(-100, -100), Point(-100, 100))) #left
        # store the lines (not considered as obstacles)
        self.lines = []
        self.rectangles = []

       
    
    # adds a new wall to the environment
    def add_wall(self, segment: Segment):
        self.walls.append(segment)
    
    # adds a new line to the environment(no impact on the physics)
    def add_line(self, segment: Segment):
        self.lines.append(segment)
    
    # adds a new rect to the environment(no impact on the physics)
    def add_rect(self, coordinate):
        self.rectangles.append(coordinate)

    # adds a new robot with initial position of 'pos' - returns the new id
    def add_robot(self, pos: Point, color: Qt.GlobalColor):
        # x and y are multipled by 20 because of the plane grid scale
        self.robots_pos = np.vstack((self.robots_pos, [pos.x, pos.y]))
        self.robots_vel = np.vstack((self.robots_vel, [0, 0]))
        self.robots_command_vel = np.vstack((self.robots_command_vel, [0, 0]))
        self.robots_color.append(color)
        return len(self.robots_pos)-1 

    # check if an id exist in the simulator (this funcion will be used in other functions in the class)
    def check_id(self, id: int):
        if id < 0 or id >= len(self.robots_pos):
            raise Exception("[ERROR] invalid robot id")

    # sends velocity to the robot number 'id' in (magnitude and direction) using 'vel'
    def robot_command(self, id: int, vel: Point):
        self.check_id(id)
        self.robots_command_vel[id, :] = [vel.x, vel.y]
    
    # returns position and velocity of robot 'id' - returns a tuple: (pos:Point, vel:Point)
    def robot_feedback(self, id: int):
        self.check_id(id)
        return (Point(self.robots_pos[id, :]), Point(self.robots_vel[id, :]))

    # moves the simulator one step in time (with respect to 'delta_t')
    def step(self):
        self.robots_vel = np.copy(self.robots_command_vel)
        self.robots_potential_dest = self.robots_pos + self.robots_vel*self.delta_t # todo check for obstacles
        robots_dir = [self.robots_vel[id]/np.linalg.norm(self.robots_vel[id], 2) if np.linalg.norm(self.robots_vel[id], 2) != 0 else 1 for id in range(len(self.robots_pos))]
        path_segs = [Segment(pos, dest+self.ROBOT_RADIUS*robots_dir[id]) for id, (pos, dest) in enumerate(zip(self.robots_pos, self.robots_potential_dest))]
        # +self.ROBOT_RADIUS*self.robots_vel[id]/np.linalg.norm(self.robots_vel[id], 2)
        intersect_points = [[intersect(path_segs[id], wall) for wall in self.walls] for id in range(len(self.robots_pos))]
        for id in range(len(self.robots_pos)):
            for intersect_point in intersect_points[id]:
                if intersect_point is not None:
                    dest = intersect_point + self.ROBOT_RADIUS*Point(self.robots_pos[id] - self.robots_potential_dest[id])/abs(Point(self.robots_pos[id] - self.robots_potential_dest[id]))
                    break
            else:
                dest = Point(self.robots_potential_dest[id])
            self.robots_pos[id] = [dest.x, dest.y]

    def removeRobots(self):
        self.robots_pos = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        self.robots_vel = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        self.robots_command_vel = np.array([], dtype=np.float32).reshape(0,2) # todo try float16 later
        self.robots_color = []

# if __name__ == '__main__':

    # seg1 = Segment((-1, 1), Point(-1, 1)+Point(1e-6, 1e-6))
    # print(seg1)
    # seg2 = Segment((0, 1), (0, -1))
    # print(intersect(seg1, seg2))
    




    # ps.add_robot(Point(2.0, 0))
    # ps.add_robot(Point(2, 1))

    # ps.robot_command(0, Point(1, 2))
    # ps.robot_command(1, Point(0, 0.2))
    # print(ps.robots_pos)
    # print(ps.robots_vel)

    # ps.step()
    # print(ps.robots_pos)
    # print(ps.robots_vel)


    