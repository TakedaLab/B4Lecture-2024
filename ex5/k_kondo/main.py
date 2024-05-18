import matplotlib.pyplot as plt
import numpy as np
import argparse


def read_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename1")
    parser.add_argument("filename2")
    parser.add_argument("filename3")
    args = parser.parse_args()

    data1 = np.loadtxt(args.filename1, delimiter=",")
    data2 = np.loadtxt(args.filename2, delimiter=",")
    data3 = np.loadtxt(args.filename3, delimiter=",")

    return data1, data2, data3


def plot_scatter(data, title):
    if data.ndim == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data, np.zeros(len(data)), marker=".")
        ax.set_xlabel("X")
        ax.set_title(title)

    elif data.ndim == 2 and data.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[:, 0], data[:, 1], marker=".")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

    elif data.ndim == 2 and data.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

    else:
        print("Data dimension not supported")

    plt.show()


def main():
    data1, data2, data3 = read_file()
    plot_scatter(data1, "data1")
    plot_scatter(data2, "data2")
    plot_scatter(data3, "data3")


if __name__ == "__main__":
    main()
