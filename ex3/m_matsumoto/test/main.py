"""Main 1 ."""

import linear_regression as mylr
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # load data
    data = np.loadtxt("../../data1.csv", delimiter=",", skiprows=1)
    x, y = data[:, 0], data[:, 1]
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)

    # print(x)
    # generate model & calcurate coefficients
    N = 1
    X = mylr.model1d(x, N)
    print(X)
    beta = mylr.calc_coef(X, y)
    # print(beta)
    # draw graph
    x_axis = np.linspace(min(x), max(x), 100)
    plt.scatter(x, y, facecolor="None", edgecolors="red", label="Observed")
    plt.plot(x_axis, mylr.expect1d(beta, x_axis), label=mylr.label1d(beta))
    plt.title("Simple Linear-Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
