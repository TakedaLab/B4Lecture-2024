"""3次元データを回転させるgifアニメーションを作成するプログラム.

できれば関数化したかったけど上手なやり方わからず
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# 三次元に投影するように設定
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# ここを書き換えて描画対象を変更
def init():
    """グラフの描画を行う関数.

    この関数を書き換えて描画対象のグラフを変更する.
    """
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
