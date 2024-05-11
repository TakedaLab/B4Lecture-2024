import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
import scipy.stats
import main

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# csvファイルの読み込み
data = np.genfromtxt("./data2.csv", delimiter=",", skip_header=1)
data = scipy.stats.zscore(data)

w,v,cr = main.calc_PCA(data)
print(v)

X,Y,Z = data[:,0],data[:,1],data[:,2]


def init():
    ax.scatter(X,Y,Z)
    main.axline3D(ax, 0, v[:,0], scale=3, color='red', linewidth=2)
    main.axline3D(ax, 0, v[:,1], scale=3, color='green', linewidth=2)
    main.axline3D(ax, 0, v[:,2], scale=3, color='blue', linewidth=2)
    return fig,

def animate(i):
    ax.view_init(elev=30., azim=3.6*i)
    return fig,

# Animate
ani = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=100, blit=True)    
ani.save('rotate.gif', writer="ffmpeg",dpi=100)
