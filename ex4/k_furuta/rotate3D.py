"""3次元データを回転させるgifアニメーションを作成するプログラム.

できれば関数化したかったけど上手なやり方わからず
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import main

# 三次元に投影するように設定
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# csvファイルの読み込み
data = np.genfromtxt("./data2.csv", delimiter=",", skip_header=1)
# データを標準化
data = scipy.stats.zscore(data)

# 主成分を求める
w, v, cr = main.calc_PCA(data)
# プロット用に軸ごとに分割
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]


# ここを書き換えて描画対象を変更
def init():
    """グラフの描画を行う関数.

    この関数を書き換えて描画対象のグラフを変更する.
    """
    ax.scatter(X, Y, Z)
    main.axline3D(ax, 0, v[:, 0], scale=3, color="red", linewidth=2)
    main.axline3D(ax, 0, v[:, 1], scale=3, color="green", linewidth=2)
    main.axline3D(ax, 0, v[:, 2], scale=3, color="blue", linewidth=2)
    return (fig,)


# ここは書き換えない
def animate(i):
    """フレームごとに呼び出される関数.

    Parameters
    -------------
    i : フレーム数
    """
    ax.view_init(elev=30.0, azim=3.6 * i)
    return (fig,)


# Animationを作成
ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=100, interval=100, blit=True
)
ani.save("rotate.gif", writer="ffmpeg", dpi=100)
