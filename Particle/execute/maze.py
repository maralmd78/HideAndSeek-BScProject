import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from copy import deepcopy

class MazeSim():
    def __init__(self):
        self.state_space = np.arange(0, 20*20, 1) # 20 cells for each robot -->20*20=400
        self.action_space = ['up', 'right', 'down', 'left']

        # construct the maze
        self.maze = np.zeros((7, 7))
        self.maze[0, :] = -1
        self.maze[-1, :] = -1
        self.maze[:, 0] = -1
        self.maze[:, -1] = -1
        self.maze[1, 3] = -1
        self.maze[2, 3] = -1
        self.maze[3, 5] = -1
        self.maze[4, [2, 4]] = -1
        
        ## the code below is just the calculation we need to do once after that we just use the resluts in a variable for a better performance
        # self.index_2_state = {}
        # self.state_2_index = {}
        # state = 0
        # for i, row in enumerate(self.maze):
        #   for j, cell in enumerate(row):
        #     if cell == -1:
        #       continue
        #     self.index_2_state[(i, j)] = state
        #     self.state_2_index[state] = (i, j)
        #     state += 1
        # print(self.index_2_state)
        # print(self.state_2_index)
        self.helper_index_2_state = {(1, 1): 0, (1, 2): 1, (1, 4): 2, (1, 5): 3, (2, 1): 4, (2, 2): 5, (2, 4): 6, (2, 5): 7, (3, 1): 8, (3, 2): 9, (3, 3): 10, (3, 4): 11, (4, 1): 12, (4, 3): 13, (4, 5): 14, (5, 1): 15, (5, 2): 16, (5, 3): 17, (5, 4): 18, (5, 5): 19}
        self.helper_state_2_index = {0: (1, 1), 1: (1, 2), 2: (1, 4), 3: (1, 5), 4: (2, 1), 5: (2, 2), 6: (2, 4), 7: (2, 5), 8: (3, 1), 9: (3, 2), 10: (3, 3), 11: (3, 4), 12: (4, 1), 13: (4, 3), 14: (4, 5), 15: (5, 1), 16: (5, 2), 17: (5, 3), 18: (5, 4), 19: (5, 5)}
        self.action_2_number = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
        
        self.state = self.index_2_state((1, 1), (4, 5))
    
    def state_2_index(self, s):
        r = s % 20
        q = s // 20   
        return (self.helper_state_2_index[q], self.helper_state_2_index[r])

    def index_2_state(self, seeker_index, hider_index):
        seeker_tmp = self.helper_index_2_state[seeker_index]
        hider_tmp = self.helper_index_2_state[hider_index]
        return 20*seeker_tmp + hider_tmp
    
    def step(self, action_seeker, action_hider):
        (i_s, j_s), (i_h, j_h) = self.state_2_index(self.state)
        
        reward_seeker = -1
        reward_hider = 1
        done = False
        
        # seeker
        if action_seeker == 'up':
            if self.maze[i_s-1, j_s] != -1:
                i_s = i_s - 1
            elif self.maze[i_s-1, j_s] == -1:
                reward_seeker -= 5 # hit the wall
        elif action_seeker == 'right':
            if self.maze[i_s, j_s+1] != -1:
                j_s = j_s + 1
            elif self.maze[i_s, j_s+1] == -1:
                reward_seeker -= 5 # hit the wall
        elif action_seeker == 'down':
            if self.maze[i_s+1, j_s] != -1:
                i_s = i_s + 1
            elif self.maze[i_s+1, j_s] == -1:
                reward_seeker -= 5 # hit the wall
        elif action_seeker == 'left':
            if self.maze[i_s, j_s-1] != -1:
                j_s = j_s - 1
            elif self.maze[i_s, j_s-1] == -1:
                reward_seeker -= 5 # hit the wall
        
        # hider
        if action_hider == 'up':
            if self.maze[i_h-1, j_h] != -1:
                i_h = i_h - 1
            elif self.maze[i_h-1, j_h] == -1:
                reward_hider -= 5 # hit the wall
        elif action_hider == 'right':
            if self.maze[i_h, j_h+1] != -1:
                j_h = j_h + 1
            elif self.maze[i_h, j_h+1] == -1:
                reward_hider -= 5 # hit the wall
        elif action_hider == 'down':
            if self.maze[i_h+1, j_h] != -1:
                i_h = i_h + 1
            elif self.maze[i_h+1, j_h] == -1:
                reward_hider -= 5 # hit the wall
        elif action_hider == 'left':
            if self.maze[i_h, j_h-1] != -1:
                j_h = j_h - 1
            elif self.maze[i_h, j_h-1] == -1:
                reward_hider -= 5 # hit the wall
        
        self.state = self.index_2_state((i_s, j_s), (i_h, j_h))
        if (i_s, j_s) == (i_h, j_h):
            reward_seeker += 100
            reward_hider -= 100
            done = True
        
        return (self.state, reward_seeker, reward_hider, done)
            
    
    # plot the maze in a given ax
    def render(self, ax, policy=[], cell_size=1.0, numbers_on_states=[]):
        num_rows = len(self.maze)
        num_cols = len(self.maze[0])
        board_x_length = cell_size*num_cols
        board_y_length = cell_size*num_rows

        ax.set_xlim(0, board_x_length)
        ax.set_ylim(0, board_y_length)
        ax.set_xticks([])
        ax.set_yticks([])

        # draw all lines
        for i in range(num_rows):
            ax.axhline(y=i*cell_size, linewidth=1, color = 'black')
        for i in range(num_cols):
            ax.axvline(x=i*cell_size, linewidth=1, color = 'black')

        # draw colored squers
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                if cell == -1:
                    ax.add_patch(Rectangle((j*cell_size, board_y_length-i*cell_size-cell_size), cell_size, cell_size, facecolor="black"))
        (i_s, j_s), (i_h, j_h) = self.state_2_index(self.state)
        ax.add_patch(Rectangle((j_s*cell_size, board_y_length-i_s*cell_size-cell_size), cell_size, cell_size, facecolor="red"))
        ax.add_patch(Rectangle((j_h*cell_size, board_y_length-i_h*cell_size-cell_size), cell_size, cell_size, facecolor="green"))

        # draw numbers (optional)
        if len(numbers_on_states) == len(self.state_space): # number of states
            for i_, num in enumerate(numbers_on_states):
                i, j = self.state_2_index[i_+1]
                ax.text(j*cell_size+cell_size/2, board_y_length-i*cell_size-cell_size+cell_size/2, str(num), rotation='horizontal', horizontalalignment='center', verticalalignment='center', color='black', fontweight='bold', fontsize='small')
            
        # draw policies (optional)
        if len(policy) == len(self.state_space): # number of states
            for i_, policy in enumerate(policy):
                i, j = self.state_2_index[i_+1]
                # x = j*cell_size+cell_size/2
                # y = board_y_length-i*cell_size-cell_size+cell_size/2
                if policy == 'up':
                    x = j*cell_size+cell_size/2
                    y = board_y_length-i*cell_size-cell_size+cell_size/4
                    dx = 0
                    dy = cell_size/2
                    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.04, color='black')
                elif policy == 'right':
                    x = j*cell_size+cell_size/4
                    y = board_y_length-i*cell_size-cell_size+cell_size/2
                    dx = cell_size/2
                    dy = 0
                    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.04, color='black')
                elif policy == 'down':
                    x = j*cell_size+cell_size/2
                    y = board_y_length-i*cell_size-cell_size+3*cell_size/4
                    dx = 0
                    dy = -cell_size/2
                    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.04, color='black')
                elif policy == 'left':
                    x = j*cell_size+3*cell_size/4
                    y = board_y_length-i*cell_size-cell_size+cell_size/2
                    dx = -cell_size/2
                    dy = 0
                    ax.arrow(x, y, dx, dy, length_includes_head=True, width=0.04, color='black')
                    
    # computes the action based on pi epsilon greedy with the given eps for the given state
    def pi_eps_greedy(self, Qtable: np.ndarray, eps: float, s: int):
        decide = np.random.choice(['greedy', 'random'], size=1, p=[1-eps, eps])[0]
        return self.action_space[np.random.choice(np.argwhere(Qtable[s-1, :] == np.amax(Qtable[s-1, :])).flatten(), size=1)[0]] if decide == 'greedy' else np.random.choice(self.action_space, size=1)[0]
        # return self.action_space[np.argmax(Qtable[s, :])] if decide == 'greedy' else np.random.choice(self.action_space, size=1)[0] # problem: chooses the first one in multiple max


def Qlearning(maze:MazeSim, Qtable_seeker_init, Qtable_hider_init, episode_num:int, max_step_num:int, gamma:float, alpha:float, epsilon=float):
    epsilon_start = 1.0
    epsilon_end = 0.001
    decay_rate = (epsilon_start - epsilon_end) / episode_num
    epsilon = epsilon_start
    
    Qtable_seeker = Qtable_seeker_init
    Qtable_hider = Qtable_hider_init
    Qtable_seeker_prev = np.copy(Qtable_seeker)
    Qtable_hider_prev = np.copy(Qtable_hider)
    rewards_seeker = np.zeros(episode_num, dtype=int)
    rewards_hider = np.zeros(episode_num, dtype=int)
    norms_seeker = np.zeros(episode_num, dtype=float)
    norms_hider = np.zeros(episode_num, dtype=float)
    # Qtable_seeker = np.zeros((len(maze.state_space), len(maze.action_space)), dtype=float)
    # Qtable_hider = np.zeros((len(maze.state_space), len(maze.action_space)), dtype=float)
    for episode in tqdm(range(episode_num)):
        maze.state = np.random.choice(maze.state_space, size=1)[0]
        action_seeker = maze.pi_eps_greedy(Qtable_seeker, epsilon, maze.state)
        action_hider = maze.pi_eps_greedy(Qtable_hider, epsilon, maze.state)
        for step in range(max_step_num):
            state = deepcopy(maze.state)
            next_state, reward_seeker, reward_hider, done = maze.step(action_seeker, action_hider)
            rewards_seeker[episode] += reward_seeker
            rewards_hider[episode] += reward_hider
            action_seeker_number = maze.action_2_number[action_seeker]
            action_hider_number = maze.action_2_number[action_hider]
            Qtable_seeker[state, action_seeker_number] = Qtable_seeker[state, action_seeker_number] + alpha * (reward_seeker + gamma*np.max(Qtable_seeker[next_state, :]) - Qtable_seeker[state, action_seeker_number])
            Qtable_hider[state, action_hider_number] = Qtable_hider[state, action_hider_number] + alpha * (reward_hider + gamma*np.max(Qtable_hider[next_state, :]) - Qtable_hider[state, action_hider_number])

            action_seeker = maze.pi_eps_greedy(Qtable_seeker, epsilon, maze.state)
            action_hider = maze.pi_eps_greedy(Qtable_hider, epsilon, maze.state)
            if done:
                break
        epsilon -= decay_rate
        Qtable_seeker_diff = Qtable_seeker - Qtable_seeker_prev
        Qtable_hider_diff = Qtable_hider - Qtable_hider_prev
        norms_seeker[episode] = np.linalg.norm(Qtable_seeker_diff, ord=2)
        norms_hider[episode] = np.linalg.norm(Qtable_hider_diff, ord=2)
        Qtable_seeker_prev = np.copy(Qtable_seeker)
        Qtable_hider_prev = np.copy(Qtable_hider)
        # epsilon = initial_epsilonn/(1+decay_rate*episode)
        
    # policy_seeker = [maze.action_space[np.argmax(row)] for row in Qtable_seeker]
    # policy_hider = [maze.action_space[np.argmax(row)] for row in Qtable_hider]
    return (Qtable_seeker, Qtable_hider, rewards_seeker, rewards_hider, norms_seeker, norms_hider)


if __name__ == '__main__':
    maze = MazeSim()
    new_state, reward_seeker, reward_hider, done = maze.step('down', 'down')
    
    policy_seeker, policy_hider = Qlearning(maze, 5000, 2000, gamma=0.95, alpha=0.3, eps=0.1)
    
    # state = maze.index_2_state((5, 5), (4, 5))
    # print(policy_seeker[state])
    # print(policy_hider[state])
    
    
    fig, ax = plt.subplots()
    maze.render(ax)
    plt.show()