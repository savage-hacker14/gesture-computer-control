# from matplotlib.animation import FuncAnimation
# import matplotlib.pyplot as plt
# import random

# # initial data
# x = [1]
# y = [random.randint(1,10)]

# # creating the first plot and frame
# fig, ax = plt.subplots()
# graph = ax.plot(x,y,color = 'g')[0]
# plt.ylim(0,10)


# # updates the data and graph
# def update(frame):
#     global graph

#     # updating the data
#     x.append(x[-1] + 1)
#     y.append(random.randint(1,10))

#     # creating a new graph or updating the graph
#     graph.set_xdata(x)
#     graph.set_ydata(y)
#     graph.set
#     plt.xlim(x[0], x[-1])

# anim = FuncAnimation(fig, update, frames = None)
# plt.show()

# -----------------------------------------------------------------------------------------------------------

# import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation
# import pandas as pd


# a = np.random.rand(2000, 3)*10
# t = np.array([np.ones(100)*i for i in range(20)]).flatten()
# df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})

# def update_graph(num):
#     data=df[df['time']==num]
#     graph.set_data (data.x, data.y)
#     graph.set_3d_properties(data.z)
#     title.set_text('3D Test, time={}'.format(num))
#     return title, graph, 


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')

# data=df[df['time']==0]
# graph, = ax.plot(data.x, data.y, data.z, linestyle="", marker="o")

# ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=100, blit=True)

# plt.show()

# -----------------------------------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

X = np.random.rand(100, 3)*10
Y = np.random.rand(100, 3)*5

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2])
fig.show()

for i in range(0, 20):
    plt.pause(1)

    Y = np.random.rand(100, 3)*5

    sc._offsets3d = (Y[:,0], Y[:,1], Y[:,2])
    plt.draw()