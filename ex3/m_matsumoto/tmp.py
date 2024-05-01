"""Read files and do linear regression."""

import csv

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """
    Load csv files.

    Args:
        path (str): A file path.

    Returns:
        ndarray: Contents of csv file.
    """
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        buf = [row for row in reader]
    # Convert to ndarray in float64
    csv_data = np.array(buf[1:])
    csv_data = csv_data.astype(np.float64)

    return csv_data


def ridge_reg(x, y, n, k):
    """
    Perform ridge regression.

    When x has 2 or more rows like [x1, x2],
    this function will return coefficients like [a,b,c,...,g]
    of a+bx1^1+cx1^2+...+dx1^n+ex2+fx2^2+...gx2^n

    Args:
        x (ndarray): An array of explanatory variables.
        y (ndarray): An array of target variables.
        n (int): Regression order.
        k (int): regularization factor.

    Returns:
        ndarray: Coefficients of polynomial in order n.
    """
    if x.ndim == 1:
        x = x.reshape(-1, len(x)).T
    N = n * x.shape[1] + 1

    # Exponent part
    j = 0
    # Row number
    row = 0
    # Create a matrix of explanatory variables and identity matrix
    poly_x = np.zeros([x.shape[0], N])
    for i in range(N):
        # Reset index number and Handle next row(When x has 2 or more rows)
        if i and i % (n + 1) == 0:
            j = 1
            row += 1
        poly_x[:, i] = x[:, row] ** j
        j += 1

    # Identity matrix in order N
    matrix_I = np.eye(N)
    # Calculate a matrix of Regression coefficients beta
    tmp = np.dot(poly_x.T, poly_x)
    tmp = np.dot(np.linalg.inv(tmp + k * matrix_I), poly_x.T)
    beta = np.dot(tmp, y)

    return beta


def calc_regression(x, beta, size):
    """
    Calculate linear regression results.

    Args:
        x (ndarray): Explanatory variables used in the regression.
        beta (ndarray): Parameters obtained in the regression.
        size (int): Size of coordinate system to define.

    Returns:
        ndarray, ndarray: Defined coordinates and regression results.
    """
    # For 2D plots
    if x.ndim == 1:
        order = len(beta) - 1
        X = np.linspace(np.min(x), np.max(x), size)
        predict = np.zeros(X.shape)
        for i in range(order + 1):
            predict += X**i * beta[i]
        return X, predict
    # For 3D plots
    elif x.ndim == 2:
        order = int((len(beta) - 1) / 2)
        X = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), size)
        Y = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), size)
        XX, YY = np.meshgrid(X, Y)
        for i in range(order + 1):
            if i == 0:
                predict = XX**i * beta[i]
            else:
                predict += XX**i * beta[i] + YY**i * beta[i + order]
        return XX, YY, predict


def main():
    """
    Perform linear regressions and plot on graphs.

    Returns:
        None
    """
    # Load csv files
    data1 = load_csv(
        r"/Users/mattsunkun/research/bm-rinko/B4Lecture-2024/ex3/data1.csv"
    )
    data2 = load_csv(
        r"/Users/mattsunkun/research/bm-rinko/B4Lecture-2024/ex3/data2.csv"
    )
    data3 = load_csv(
        r"/Users/mattsunkun/research/bm-rinko/B4Lecture-2024/ex3/data3.csv"
    )

    # Liniear regression
    x1 = data1[:, 0]
    y1 = data1[:, 1]
    beta1 = ridge_reg(x1, y1, 1, 0)
    x1p, y1_predict = calc_regression(x1, beta1, 2000)
    str1 = f"{beta1[1]:+.2f}x{beta1[0]:+.2f}"

    print(beta1)
    exit()

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    beta2 = ridge_reg(x2, y2, 3, 0)
    x2p, y2_predict = calc_regression(x2, beta2, 2000)
    str21 = f"{beta2[3]:+.2f}x^3{beta2[2]:+.2f}"
    str22 = f"x^2{beta2[1]:+.2f}x{beta2[0]:+.2f}"

    x3 = data3[:, 0]
    y3 = data3[:, 1]
    z3 = data3[:, 2]
    beta3 = ridge_reg(data3[:, :2], z3, 2, 0)
    XX, YY, ZZ = calc_regression(data3[:, :2], beta3, 10)
    str31 = f"{beta3[2]:+.2f}x^2{beta3[1]:+.2f}x{beta3[0]:+.2f}"
    str32 = f"{beta3[3]:+.2f}y^2{beta3[4]:+.2f}y"

    # Plot data
    plt.title("data1.csv")
    plt.scatter(x1, y1, marker=".", color="blue", label="data1")
    plt.plot(x1p, y1_predict, color="orange", label=str1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    plt.title("data2.csv")
    plt.scatter(x2, y2, marker=".", color="blue", label="data2")
    plt.plot(x2p, y2_predict, color="orange", label=str21 + str22)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("data3.csv")
    ax.scatter(x3, y3, z3, marker=".", color="blue", label="data3")
    ax.plot_wireframe(XX, YY, ZZ, color="orange", label=str31 + str32)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    exit(1)
