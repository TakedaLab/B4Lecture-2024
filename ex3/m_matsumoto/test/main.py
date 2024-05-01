"""Main 3."""

import argparse

# import japanize_matplotlib
import linear_regression as mylr
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # args parser settings
    parser = argparse.ArgumentParser(description="Process linear-regression")
    parser.add_argument("-a", "--alpha", type=np.float64, default=1)

    # get args
    args = parser.parse_args()
    alpha = args.alpha

    # load data
    data = np.loadtxt("../../data3.csv", delimiter=",", skiprows=1)
    x1, x2, y = data[:, 0], data[:, 1], data[:, 2]
    x1, x2, y = x1.reshape(-1, 1), x2.reshape(-1, 1), y.reshape(-1, 1)

    # calculate coefficients
    N1 = 1
    N2 = 2
    X = mylr.model2d(x1, x2, N1, N2)
    beta = mylr.calc_coef(X, y)
    beta_with_regular = mylr.calc_coef_with_regularization(X, y, alpha)

    # draw graph
    x1_axis = np.linspace(min(x1), max(x2), 100)
    x2_axis = np.linspace(min(x2), max(x2), 100)
    # print(x1_axis)
    # print(x2_axis)
    # exit()
    x1_mesh, x2_mesh = np.meshgrid(x1_axis, x2_axis)
    y_expect = mylr.expect2d(beta, N1, N2, x1_mesh, x2_mesh)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlabel("$x_{1}$", fontsize=22)
    ax.set_ylabel("$x_{2}$", fontsize=22)
    ax.set_zlabel("y", fontsize=22)
    ax.scatter(x1, x2, y, marker="o", c="red", label="Observed data")
    ax.plot_wireframe(
        x1_mesh,
        x2_mesh,
        mylr.expect2d(beta, N1, N2, x1_mesh, x2_mesh),
        color="blue",
        label=mylr.label2d(beta, N1, N2),
    )
    ax.legend()

    ax = fig.add_subplot(122, projection="3d")
    ax.set_xlabel("$x_{1}$", fontsize=22)
    ax.set_ylabel("$x_{2}$", fontsize=22)
    ax.set_zlabel("y", fontsize=22)
    ax.scatter(x1, x2, y, marker="o", c="red", label="Observed data")
    ax.plot_wireframe(
        x1_mesh,
        x2_mesh,
        mylr.expect2d(beta_with_regular, N1, N2, x1_mesh, x2_mesh),
        color="orange",
        label=mylr.label2d(beta_with_regular, N1, N2) + "(正則化)",
    )
    ax.legend()
    plt.tick_params(labelsize=15)
    plt.show()
