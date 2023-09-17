import pickle
import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    return np.hstack((np.linspace(0, ret[0], n-1), ret))


with open('realposition(right move)', "rb") as f:  
    positions = pickle.load(f)

positions = positions[0]
positions_x = []
positions_y = []
for position in positions:
    positions_x.append(position['x'])
    positions_y.append(position['x'])
# print(positions[0]['x'][0:30])
## the center of real world
setpoint_x = [60]*50
setpoint_y = [60]*50
y = list(range(1, len(positions) + 1))
#### 2D plot without scatter
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

## position plot
ax.set_title("position tracking (settpoint=(60, 60), center)")
ax.plot(y, positions_x, label='position_x', color='C0')
# ax.plot(y, positions_y, label='position_y', color='C1')

ax.plot(y, setpoint_x, label='setpoint_x', color='C2')
# ax.plot(y, setpoint_y, label='setpoint_y', color='C3')

ax.set_xlabel("Time(ms)")
ax.set_ylabel("Robot's position(m)")
ax.legend()


## error plot
# ax.set_title("error (setpoint=(20, 20))")
# ax.plot(times[0:30], errors[0:30], color='C4')

# ax.set_xlabel("Time(sec)")
# ax.set_ylabel("Error")



#### scatter plot without scatter for three setpoints: (30, 0) (0, 40) (20, 20)
# plt.figure(figsize=(8, 8))
# sc = plt.scatter(positions_x[0:30], positions_y[0:30], c=times[0:30], cmap='viridis', marker='o')

# # Add colorbar to show time progression
# cbar = plt.colorbar(sc)
# cbar.set_label('Time(sec)')

# # Customize plot settings
# plt.title('position tracking (settpoint=(0, 40))')
# plt.xlabel('position_x(m)')
# plt.ylabel('position_y(m)')

plt.show()