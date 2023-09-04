import pickle
import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    return np.hstack((np.linspace(0, ret[0], n-1), ret))


with open('Qtable_train_seekerOnly12_(maze, 0, 0, 220000, 1000, 0.99, 0.2).pkl', "rb") as f:  
    Qtable_seeker, Qtable_hider, rewards_seeker, rewards_hider = pickle.load(f)

data = rewards_hider
window = 700
rewards_data_movingAve = moving_average(data, n=window)
episodes = np.arange(0, len(data), 1, dtype=int)
episodes_movingAve = episodes[-len(rewards_data_movingAve):]
### 2D plot - seeker rewards
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

ax.set_title("Progress of the hider agent when training the seeker agent")
ax.plot(episodes, data, label='Hider (cumulative)', color="silver")
ax.plot(episodes_movingAve, rewards_data_movingAve, label=f'Hider (moving average - w={window})', color="red", linewidth=2)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_xlim((episodes.min(), episodes.max()))
ax.set_ylim((data.min(), data.max()))
ax.legend()
plt.savefig("trainingSeeker-hiderplot.jpg", dpi=500)
plt.show()








exit(0)
### controller plot
with open('error_pos(0, 40).pkl', "rb") as f:  
    errors, positions, times, setpoint = pickle.load(f)
positions_x = [item[0] for item in positions]
positions_y = [item[1] for item in positions]
setpoint_x = setpoint.x
setpoints_x = [setpoint_x]*len(times)

setpoint_y = setpoint.y
setpoints_y = [setpoint_y]*len(times)

#### 2D plot without scatter for three setpoints: (30, 0) (0, 40) (20, 20)
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

## position plot
# ax.set_title("position tracking (settpoint=(20, 20))")
# ax.plot(times[0:30], positions_x[0:30], label='position_x', color='C0')
# ax.plot(times[0:30], positions_y[0:30], label='position_y', color='C1')

# ax.plot(times[0:30], setpoints_x[0:30], label='setpoint_x', color='C2')
# ax.plot(times[0:30], setpoints_y[0:30], label='setpoint_y', color='C3')

# ax.set_xlabel("Time(sec)")
# ax.set_ylabel("Robot's position(m)")
# ax.legend()


## error plot
# ax.set_title("error (setpoint=(20, 20))")
# ax.plot(times[0:30], errors[0:30], color='C4')

# ax.set_xlabel("Time(sec)")
# ax.set_ylabel("Error")



#### scatter plot without scatter for three setpoints: (30, 0) (0, 40) (20, 20)
plt.figure(figsize=(8, 8))
sc = plt.scatter(positions_x[0:30], positions_y[0:30], c=times[0:30], cmap='viridis', marker='o')

# Add colorbar to show time progression
cbar = plt.colorbar(sc)
cbar.set_label('Time(sec)')

# Customize plot settings
plt.title('position tracking (settpoint=(0, 40))')
plt.xlabel('position_x(m)')
plt.ylabel('position_y(m)')

plt.show()