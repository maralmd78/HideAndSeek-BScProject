import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import pickle

class Color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'


class MazeSim():
  def __init__(self, p=0.1):
    self.p = p
    self.state_space = np.arange(1, 77, 1)
    self.action_space = policy_space = ['up', 'right', 'down', 'left']
    
    # construct the maze
    self.maze = np.zeros((12, 12))
    self.maze[0, :] = -1
    self.maze[-1, :] = -1
    self.maze[:, 0] = -1
    self.maze[:, -1] = -1
    self.maze[1, 2] = -1
    self.maze[2, [2, 4, 5, 6, 7, 8]] = -1
    self.maze[3, [2, 8]] = -1
    self.maze[4, [6, 8]] = -1
    self.maze[5, [4, 5]] = -1
    self.maze[7, [1, 2, 5, 8]] = -1
    self.maze[8, [5, 6, 7, 8]] = -1
    self.maze[9, [3, 4, 10]] = -1
    
    # print(self.maze)

    ## the code below is just the calculation we need to do once after that we just use the resluts in a variable for a better performance
    # self.index_2_state = {}
    # self.state_2_index = {}
    # state = 1
    # for i, row in enumerate(self.maze):
    #   for j, cell in enumerate(row):
    #     if cell == -1:
    #       continue
    #     self.index_2_state[(i, j)] = state
    #     self.state_2_index[state] = (i, j)
    #     state += 1
    # print(self.index_2_state)
    # print(self.state_2_index)
    self.index_2_state = {(1, 1): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 5, (1, 7): 6, (1, 8): 7, (1, 9): 8, (1, 10): 9, (2, 1): 10, (2, 3): 11, (2, 9): 12, (2, 10): 13, (3, 1): 14, (3, 3): 15, (3, 4): 16, (3, 5): 17, (3, 6): 18, (3, 7): 19, (3, 9): 20, (3, 10): 21, (4, 1): 22, (4, 2): 23, (4, 3): 24, (4, 4): 25, (4, 5): 26, (4, 7): 27, (4, 9): 28, (4, 10): 29, (5, 1): 30, (5, 2): 31, (5, 3): 32, (5, 6): 33, (5, 7): 34, (5, 8): 35, (5, 9): 36, (5, 10): 37, (6, 1): 38, (6, 2): 39, (6, 3): 40, (6, 4): 41, (6, 5): 42, (6, 6): 43, (6, 7): 44, (6, 8): 45, (6, 9): 46, (6, 10): 47, (7, 3): 48, (7, 4): 49, (7, 6): 50, (7, 7): 51, (7, 9): 52, (7, 10): 53, (8, 1): 54, (8, 2): 55, (8, 3): 56, (8, 4): 57, (8, 9): 58, (8, 10): 59, (9, 1): 60, (9, 2): 61, (9, 5): 62, (9, 6): 63, (9, 7): 64, (9, 8): 65, (9, 9): 66, (10, 1): 67, (10, 2): 68, (10, 3): 69, (10, 4): 70, (10, 5): 71, (10, 6): 72, (10, 7): 73, (10, 8): 74, (10, 9): 75, (10, 10): 76}
    self.state_2_index = {1: (1, 1), 2: (1, 3), 3: (1, 4), 4: (1, 5), 5: (1, 6), 6: (1, 7), 7: (1, 8), 8: (1, 9), 9: (1, 10), 10: (2, 1), 11: (2, 3), 12: (2, 9), 13: (2, 10), 14: (3, 1), 15: (3, 3), 16: (3, 4), 17: (3, 5), 18: (3, 6), 19: (3, 7), 20: (3, 9), 21: (3, 10), 22: (4, 1), 23: (4, 2), 24: (4, 3), 25: (4, 4), 26: (4, 5), 27: (4, 7), 28: (4, 9), 29: (4, 10), 30: (5, 1), 31: (5, 2), 32: (5, 3), 33: (5, 6), 34: (5, 7), 35: (5, 8), 36: (5, 9), 37: (5, 10), 38: (6, 1), 39: (6, 2), 40: (6, 3), 41: (6, 4), 42: (6, 5), 43: (6, 6), 44: (6, 7), 45: (6, 8), 46: (6, 9), 47: (6, 10), 48: (7, 3), 49: (7, 4), 50: (7, 6), 51: (7, 7), 52: (7, 9), 53: (7, 10), 54: (8, 1), 55: (8, 2), 56: (8, 3), 57: (8, 4), 58: (8, 9), 59: (8, 10), 60: (9, 1), 61: (9, 2), 62: (9, 5), 63: (9, 6), 64: (9, 7), 65: (9, 8), 66: (9, 9), 67: (10, 1), 68: (10, 2), 69: (10, 3), 70: (10, 4), 71: (10, 5), 72: (10, 6), 73: (10, 7), 74: (10, 8), 75: (10, 9), 76: (10, 10)}
    self.action_to_index = {'up': 0, 'right': 1, 'down': 2, 'left': 3}
    self.START = 1
    self.GOAL = 58
    self.maze[self.state_2_index[self.START]] = 3
    self.maze[self.state_2_index[self.GOAL]] = 4


  # returns the neighbors of s(s=1,2,...,76) and return res that is a list(of states) of all neighbors
  def neighbors(self, s:int):
    i, j = self.state_2_index[s]
    i_up, j_up = i-1, j
    i_right, j_right = i, j+1
    i_down, j_down = i+1, j
    i_left, j_left = i, j-1

    res = [s] # the state itself is also included
    for ni, nj in [(i_up, j_up), (i_right, j_right), (i_down, j_down), (i_left, j_left)]:
      if self.maze[ni, nj] != -1:
        res.append(self.index_2_state[(ni, nj)])
    return res


  # computes the reward function: R(s, a, ns), s-->state, a-->action, ns-->nextstate
  def reward(self, s, a, ns):
    i, j = self.state_2_index[s]
    ni, nj = self.state_2_index[ns]

    if self.maze[i, j] == 4:
      return 0

    r = -1
    if self.maze[ni, nj] == 4: # goal
      r += 100 
    return r

  # computes the transition probability: p(ns|s, a)
  def transition(self, s, a, ns):
    i, j = self.state_2_index[s]
    ni, nj = self.state_2_index[ns]
    i_up, j_up = i-1, j
    i_right, j_right = i, j+1
    i_down, j_down = i+1, j
    i_left, j_left = i, j-1

    pr = 0
    if ns == s: # ns: staying
      if self.maze[i_up, j_up] == -1:
        pr += 1-self.p if a == 'up' else self.p/3
      if self.maze[i_right, j_right] == -1:
        pr += 1-self.p if a == 'right' else self.p/3
      if self.maze[i_down, j_down] == -1:
        pr += 1-self.p if a == 'down' else self.p/3
      if self.maze[i_left, j_left] == -1:
        pr += 1-self.p if a == 'left' else self.p/3
      return pr

    if (ni, nj) == (i-1, j): # ns: upper state
      return 1-self.p if a == 'up' else self.p/3

    if (ni, nj) == (i, j+1): # ns: right state
      return 1-self.p if a == 'right' else self.p/3

    if (ni, nj) == (i+1, j): # ns: down state
      return 1-self.p if a == 'down' else self.p/3

    if (ni, nj) == (i, j-1): # ns: left state
      return 1-self.p if a == 'left' else self.p/3
    return 0
  

  # find the path from the start cell untill we bump into a wall or reach the goal
  def find_final_path(self, policy):
    path = ['none' for _ in range(len(self.state_space))]
    i, j = self.state_2_index[self.START] # start point
    visited = [] # states visited
    while True:
      state = self.index_2_state[i, j]
      visited.append(state)
      state = state-1
      path[state] = policy[state]
      if policy[state] == 'up':
        i = i-1
      elif policy[state] == 'right':
        j = j+1
      elif policy[state] == 'down':
        i = i+1
      elif policy[state] == 'left':
        j = j-1
      if self.maze[i, j] == -1 or (self.index_2_state[(i, j)] in visited) or self.maze[i, j] == 4: # until bump into wall or get into a loop or reach the end
        return path
    
  # check if a given policy reaches to the goal
  def is_path_reach_goal(self, policy):
    path = self.find_final_path(policy)
    return path[52-1]=='down' or path[59-1]=='left' or path[66-1]=='up'

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
        elif cell == 3:
          ax.add_patch(Rectangle((j*cell_size, board_y_length-i*cell_size-cell_size), cell_size, cell_size, facecolor="green"))
        elif cell == 4:
          ax.add_patch(Rectangle((j*cell_size, board_y_length-i*cell_size-cell_size), cell_size, cell_size, facecolor="red"))

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

  # takes the current state and the action and returns the next state(int) and the immediate reward in a tuple: (nextstate, r)
  def step(self, s_current: int, a: str):
    neighbors = self.neighbors(s_current)
    probablities = [self.transition(s_current, a, neighbor) for neighbor in neighbors]
    s_next = np.random.choice(neighbors, size=1, p=probablities)[0]
    return (s_next, self.reward(s_current, a, s_next))

  # computes the action based on pi epsilon greedy with the given eps for the given state
  def pi_eps_greedy(self, Qtable: np.ndarray, eps: float, s: int):
    decide = np.random.choice(['greedy', 'random'], size=1, p=[1-eps, eps])[0]
    return self.action_space[np.random.choice(np.argwhere(Qtable[s-1, :] == np.amax(Qtable[s-1, :])).flatten(), size=1)[0]] if decide == 'greedy' else np.random.choice(self.action_space, size=1)[0]
    # return self.action_space[np.argmax(Qtable[s-1, :])] if decide == 'greedy' else np.random.choice(self.action_space, size=1)[0]
  
  # computes the action based on pi boltzman with the given Htable for the given state
  # returns a tuple (a:str, p:list) 'a' can be up, right, ... and 'p' is the probability of each action: {p(up), p(right), p(down), p(left)}
  def pi_boltzman(self, Htable: np.ndarray, s: int):
    p = np.array([np.exp(h) for h in Htable[s-1, :]])/np.sum(np.exp(Htable[s-1, :]))
    return (np.random.choice(self.action_space, size=1, p=p)[0], p)



def SARSA(maze_sim:MazeSim, episode_num:int, max_step_num:int, gamma:float, alpha:float, eps:float):
  Qtable = np.zeros((len(maze_sim.state_space), len(maze_sim.action_space)), dtype=float)
  all_rewards = []
  # hint: s:current state - r:immediate reward - a:current action - ns:next state - na:next action
  for episode in tqdm(range(episode_num)):
    rewards = []
    # s = maze_sim.START
    # start from a random state
    s = np.random.choice(maze_sim.state_space, size=1)[0]
    a = maze_sim.pi_eps_greedy(Qtable, eps, s)
    for step in range(max_step_num):
      # print(episode, '->', step)
      ns, r = maze_sim.step(s, a)
      na = maze_sim.pi_eps_greedy(Qtable, eps, ns)
      s_, ns_ = s-1, ns-1 # row numbers in Qtabl start from 0, but the states start from 1
      a_, na_ = maze_sim.action_to_index[a], maze_sim.action_to_index[na] # convert actions from up, right, ... to column numbers
      Qtable[s_, a_] = Qtable[s_, a_] + alpha * (r + gamma*Qtable[ns_, na_] - Qtable[s_, a_])
      s, a = ns, na
      rewards.append(r)
      if s == maze_sim.GOAL:
        break
    all_rewards.append(np.average(rewards))
  policy = [maze_sim.action_space[np.argmax(row)] for row in Qtable]
  policy[maze_sim.GOAL-1] = "None"
  return (Qtable, policy, all_rewards)



def QLearning(maze_sim:MazeSim, episode_num:int, max_step_num:int, gamma:float, alpha:float, eps:float):
  Qtable = np.zeros((len(maze_sim.state_space), len(maze_sim.action_space)), dtype=float)
  all_rewards = []
  # hint: s:current state - r:immediate reward - a:current action - ns:next state
  for episode in tqdm(range(episode_num)):
    rewards = []
    # s = maze_sim.START
    # start from a random state
    s = np.random.choice(maze_sim.state_space, size=1)[0]
    a = maze_sim.pi_eps_greedy(Qtable, eps, s)
    for step in range(max_step_num):
      # print(episode, '->', step)
      ns, r = maze_sim.step(s, a)
      s_, ns_ = s-1, ns-1 # row numbers in Qtabl start from 0, but the states start from 1
      a_ = maze_sim.action_to_index[a] # convert actions from up, right, ... to column numbers
      Qtable[s_, a_] = Qtable[s_, a_] + alpha * (r + gamma*np.max(Qtable[ns_, :]) - Qtable[s_, a_])
      s = ns
      a = maze_sim.pi_eps_greedy(Qtable, eps, s)
      rewards.append(r)
      if s == maze_sim.GOAL:
        break
    all_rewards.append(np.average(rewards))
  policy = [maze_sim.action_space[np.argmax(row)] for row in Qtable]
  policy[maze_sim.GOAL-1] = "None"
  return (Qtable, policy, all_rewards)


if __name__ == "__main__":
  with open('results/qlearning_fixed_goal/qlearning_fix_goal.pkl', 'rb') as f:
    qtable_ql, policy_ql, acc_reward_ql = pickle.load(f)
  with open('results/sarsa_fixed_goal/sarsa_fix_goal.pkl', 'rb') as f:
    qtable_sarsa, policy_sarsa, acc_reward_sarsa = pickle.load(f)

  
  fig, ax = plt.subplots()
  ax.plot(acc_reward_sarsa, color='red', linestyle='-', linewidth=2, markersize=12, label='SARSA')
  ax.plot(acc_reward_ql, color='green', linestyle='-', linewidth=2, markersize=12, label='QLearning')
  ax.legend()
  plt.show()
    
  

  # maze_sim = MazeSim(p=0.02)
  # policy_star_all = []
  # acc_rewards_all = []
  # is_reach_goal_all = []
  # for i in range(5):
  #   Qtable, policy_star, acc_rewards = SARSA(maze_sim, 1000, 1000, gamma=0.95, alpha=0.3, eps=0.1)
  #   policy_star_all.append(policy_star)
  #   acc_rewards_all.append(acc_rewards)
  #   is_reach_goal_all.append(maze_sim.is_path_reach_goal(policy_star))

  # policy_star = policy_star_all[np.argmax(is_reach_goal_all)] # the first trial reached to goal from 10 independent runs
  # acc_rewards_all = np.array(acc_rewards_all)
  # avg_acc_rewards_qlearning = np.cumsum(acc_rewards_all, axis=1) / np.arange(1, acc_rewards_all.shape[1]+1)[None, :]
  # avg_acc_rewards_qlearning = np.average(np.array(avg_acc_rewards_qlearning), axis=0)

  # fig = plt.figure(1)
  # ax = fig.add_axes([0.00, 0.00, 1.00, 1.00])
  # ax.set_title("SARSA policy", fontsize='x-large', fontweight='extra bold')
  # # maze_sim.render(ax, numbers_on_states=maze_sim.state_space)

  # maze_sim.render(ax, policy=policy_star)

  # fig, ax = plt.subplots()
  # ax.plot(avg_acc_rewards_qlearning, color='red', linestyle='-', linewidth=2, markersize=12)



  # with open('sarsa_fix_goal.pkl', 'wb') as f:
  #   pickle.dump([Qtable, policy_star, avg_acc_rewards_qlearning], f)

  # plt.show()
    