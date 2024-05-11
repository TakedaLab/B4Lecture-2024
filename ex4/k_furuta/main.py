import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import scipy.stats

def calc_PCA(data:np.ndarray):
    # 分散共分散行列を計算
    cov_matrix = np.cov(data.T)
    # 固有値と固有ベクトルを計算
    eigen_value, eigen_vector = np.linalg.eig(np.array(cov_matrix))
    # 寄与率を計算
    contribution_rate = eigen_value / np.sum(eigen_value)
    # 行ごとにソートされるため転地
    eigen_vector = eigen_vector.T
    # 寄与率が高い順にソート
    zip_lists = zip(contribution_rate, eigen_value, eigen_vector)
    zip_sort = sorted(zip_lists, reverse=True)
    # ソート済みのものから取り出す
    contribution_rate, eigen_value, eigen_vector = zip(*zip_sort)
    eigen_vector = np.array(eigen_vector).T

    return eigen_value, eigen_vector, contribution_rate


def axline3D(ax, mean, component, scale, **kwargs):
    line = np.array([mean + component * scale, mean - component * scale])
    ax.plot(line[:, 0], line[:, 1], line[:, 2], **kwargs)


def plot_PCA(data, eigen_vector, contribution_rate, file_name):
    if data.shape[1]==2:
        fig, ax = plt.subplots()
        ax.scatter(data[:,0],data[:,1])
        ax.axline((0,0),v[0], color='red', label=f'ratio:{round(contribution_rate[0],3)}')
        ax.axline((0,0),v[1], color='green', label=f'ratio:{round(contribution_rate[1],3)}')
        ax.legend()
        plt.savefig(f"Result_{file_name}.png")
    elif data.shape[1]==3:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(data[:,0],data[:,1],data[:,2])
        axline3D(ax, 0, eigen_vector[:,0], scale=3, color='red', linewidth=2, label=f'ratio:{round(contribution_rate[0],3)}')
        axline3D(ax, 0, eigen_vector[:,1], scale=3, color='green', linewidth=2, label=f'ratio:{round(contribution_rate[1],3)}')
        axline3D(ax, 0, eigen_vector[:,2], scale=3, color='blue', linewidth=2, label=f'ratio:{round(contribution_rate[2],3)}')
        ax.legend()
        plt.savefig(f"Result_{file_name}.png")
    else:
        pass


def plot_CCR(contribution_rate, file_name): 
    accumulated_rate = np.add.accumulate(contribution_rate)
    fig, ax = plt.subplots()
    ax.plot(accumulated_rate)
    ax.set_ylim(0,1.1)
    plt.savefig(f"ratio_{file_name}.png")


def parse_args():
    """コマンドプロントから引数を受け取るための関数."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Name of input csv file"
    )
    return parser.parse_args()

if __name__=="__main__":
    # 引数の受け取り
    args = parse_args()
    file_path = args.input_file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # csvファイルの読み込み
    data = np.genfromtxt(file_path, delimiter=",")
    data = scipy.stats.zscore(data)

    # 主成分分析の計算
    w,v,cr = calc_PCA(data)
    print("contribution-rate:",cr)

    # 主成分のプロット
    plot_PCA(data, v, cr, file_name)

    # 累積寄与率のプロット
    plot_CCR(cr, file_name)

    # 2次元に変換してプロット
    transformed_data = data @ v
    fig, ax = plt.subplots()
    plt.scatter(transformed_data[:,0],transformed_data[:,1])
    plt.savefig(f"transformed_{file_name}.png")

