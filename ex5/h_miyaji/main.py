"""This module performs a fitting of the data using the EM algorithm."""

import argparse

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import ex3


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="performs a fitting of tha data.")

    # csv file name
    parser.add_argument("--input-file", type=str, required=True, help="input csv file")

    # number of clusters for Gaussian distribution
    parser.add_argument("--cluster-num", type=int, default=1, help="number of clusters")

    # convergence condition for EM algorithm
    parser.add_argument(
        "--error", type=float, default=0.001, help="convergence condition"
    )

    return parser.parse_args()


def initialize_gmm(
    dim: int, cluster_num: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the means, covariance matrix and weights.

    Args:
        dim (int): the dimension of csv data.
        cluster_num (int): number of clusters.

    Returns:
        means (np.ndarray): mean for each cluster (k, dim).
        cov_matrix (np.ndarray): covariance matrix for each cluster (k, dim, dim).
        weights (np.ndarray): weight of each Gaussian (k, ).
    """
    means = np.random.randn(cluster_num, dim)  # (k, dim)
    cov_matrix = np.array([np.eye(dim) for k in range(cluster_num)])  # (k, dim, dim)
    weights = np.array([1 / cluster_num for k in range(cluster_num)])  # (k, )
    return means, cov_matrix, weights


def get_gauss(data: np.ndarray, mean: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Get the value of the Gaussian distribution.

    Args:
        data (np.ndarray): one of the csv data (dim,).
        mean (np.ndarray): mean in one cluster (dim,).
        cov_matrix (np.ndarray): covariance matrix in one cluster (dim, dim).

    Returns:
        gauss (float): the value of the Gaussian distribution.
    """
    # dimension of the data
    if isinstance(data, np.ndarray):
        dim = len(data)
    else:
        dim = 1

    gauss_denominator = (np.sqrt(2 * np.pi) ** dim) * np.sqrt(np.linalg.det(cov_matrix))
    gauss_numerator = np.exp(
        ((data - mean) @ np.linalg.inv(cov_matrix) @ (data - mean)[:, np.newaxis]) / -2
    )

    gauss = gauss_numerator.item() / gauss_denominator.item()
    return gauss


def calc_mix_gauss(
    dataset: np.ndarray,
    means: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Calculate values of mix Gaussian distribution.

    Args:
        dataset (np.ndarray): csv data (n, dim).
        means (np.ndarray): mean for each cluster (k, dim).
        cov_matrix (np.ndarray): covariance matrix for each cluster (k, dim, dim).
        weights (np.ndarray): weight of each Gaussian (k, ).

    Returns:
        mix_gauss (np.ndarray): values of mix Gaussian distribution (k, n).
    """
    cluster_num = len(weights)  # number of Gaussian
    data_num = len(dataset)  # number of csv data

    mix_gauss = np.zeros((cluster_num, data_num))  # (k, n)

    # for k in range(cluster_num):
    #     for n in range(data_num):
    #         mix_gauss[k, n] = weights.item(k) * get_gauss(
    #             dataset[n], means[k], cov_matrix[k]
    #         )

    # remove double for loop
    gaussian_pdfs = list(
        map(lambda mean, cov: multivariate_normal(mean, cov), means, cov_matrix)
    )
    mix_gauss = np.array(
        list(map(lambda pdf, weight: pdf.pdf(dataset) * weight, gaussian_pdfs, weights))
    )

    return mix_gauss


def em_algo(
    dataset: np.ndarray,
    means: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update parameters with EM algorithm.

    Args:
        dataset (np.ndarray): csv data (n, dim).
        means (np.ndarray): mean for each cluster (k, dim).
        cov_matrix (np.ndarray): covariance matrix for each cluster (k, dim, dim).
        weights (np.ndarray): weight of each Gaussian (k, ).

    Returns:
        new_means (np.ndarray): updated means (k, dim).
        new_cov_matrix (np.ndarray): updated cov_matrix (k, dim, dim).
        new_weights (np.ndarray): updated weights (k, ).
    """
    cluster_num = len(weights)  # number of Gaussian
    data_num = len(dataset)  # number of csv data
    dim = len(dataset[0])  # dimension of the data

    # E step : calculate burden ratio (gamma)
    mix_gauss = calc_mix_gauss(dataset, means, cov_matrix, weights)
    burden_ratio = mix_gauss.T / np.sum(mix_gauss, axis=0)[:, np.newaxis]  # (n, k)

    # M step : update θ = {μ, Σ, π}
    N_k = np.sum(burden_ratio, axis=0)  # (k, )
    N = np.sum(N_k)  # N = data_num (float)
    new_weights = N_k / N

    new_means = np.zeros((cluster_num, dim))
    for k in range(cluster_num):
        # (dim, ) = np.sum( (1, dim)*(n, dim), axis=0 ) = np.sum( (n, dim), axis=0 )
        new_means[k, :] = (
            np.sum(burden_ratio[:, k][:, np.newaxis] * dataset, axis=0) / N_k[k]
        )

    new_cov_matrix = np.zeros((cluster_num, dim, dim))

    # (k, n, dim) = x(1, n, dim) - μ(k, 1, dim)
    deviation = dataset[np.newaxis, :, :] - new_means[:, np.newaxis, :]

    for k in range(cluster_num):
        for n in range(data_num):
            tmp = deviation[k, n, :][:, np.newaxis] @ deviation[k, n, :][np.newaxis, :]
            new_cov_matrix[k, :, :] += burden_ratio[n, k] * tmp
        new_cov_matrix[k, :, :] /= N_k[k]

    return new_means, new_cov_matrix, new_weights


def get_log_likelihood(
    dataset: np.ndarray,
    means: np.ndarray,
    cov_matrix: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Get the value of log-likelihood.

    Args:
        dataset (np.ndarray): csv data (n, dim).
        means (np.ndarray): mean for each cluster (k, dim).
        cov_matrix (np.ndarray): covariance matrix for each cluster (k, dim, dim).
        weights (np.ndarray): weight of each Gaussian (k, ).

    Returns:
        log_likelihood (float): the value of log-likelihood.
    """
    log_likelihood = np.sum(
        np.log(np.sum(calc_mix_gauss(dataset, means, cov_matrix, weights), axis=0))
    )
    return log_likelihood


def plot_one_line(
    dataset: np.ndarray,
    title: str,
    xlabel: str = "x1",
    ylabel: str = "x2",
    zlabel: str = "x3",
) -> None:
    """Plot only one graph.

    Args:
        dataset (np.ndarray): data of x, y (and z) (n, dim).
        title (str): the graph title.
        xlabel (str, optional): label name on x-axis. Defaults to "x1".
        ylabel (str, optional): label name on y-axis. Defaults to "x2".
        zlabel (str, optional): label name on z-axis. Defaults to "x3".

    Returns:
        None:
    """
    dim = len(dataset[0])  # dimension of the data

    fig = plt.figure()

    x1 = dataset[:, 0]
    x2 = dataset[:, 1]

    if dim == 1 or dim == 2:  # x-y
        ax = fig.add_subplot(111)
        ax.plot(x1, x2, color="b")

    elif dim == 3:  # x-y-z
        x3 = dataset[:, 2]
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x1, x2, x3, color="b")
        ax.set_zlabel(zlabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    # plt.savefig("h_miyaji\\figs\\result3-2.png")
    plt.show()

    return None


def plot_gmm(
    mix_gauss: np.ndarray,
    means: np.ndarray,
    x_value: np.ndarray,
    axis_range: list,
    ax,
):
    """Plot GMM on a scatter plot.

    Args:
        mix_gauss (np.ndarray): the data of GMM (n, )
        means (np.ndarray): mean for each cluster (k, dim).
        x_value (np.ndarray): the data of x1 (and x2) axis (plot_num, dim).
        axis_range (list): the range of graph axis ([xmin, xmax, ymin, ymax]).
        ax (Axes): Axes object of the scatter plot.

    Returns:
        ax: Axes object of the graph.
    """
    # dimension of the data
    if isinstance(means[0], np.ndarray):
        dim = len(means[0])
    else:
        dim = 1
    colors = ("b", "r")  # color of plot (blue, red)

    if dim == 1:  # x
        ax.plot(x_value, mix_gauss, label="GMM", color=colors[0])
        ax.scatter(
            means,
            np.zeros((len(means))),
            label="Centroids",
            color=colors[1],
            marker="x",
            s=200,
        )

    elif dim == 2:  # x-y
        contour = ax.contour(x_value[0], x_value[1], mix_gauss, cmap="magma")
        ax.scatter(
            means[:, 0],
            means[:, 1],
            label="Centroids",
            color=colors[1],
            marker="x",
            s=200,
        )

        # create continuous colorbar
        norm = matplotlib.colors.Normalize(contour.cvalues.min(), contour.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=contour.cmap)
        sm.set_array([])
        plt.colorbar(sm, ticks=contour.levels, ax=ax, label="Probability density")

    ax.legend()
    ax.axis(axis_range)
    # plt.savefig("h_miyaji\\figs\\result3-1.png")
    return ax


def main():
    """Load csv file and perform a fitting of the data using the EM algorithm."""
    # get argument
    args = parse_args()
    filename = args.input_file  # csv file name
    cluster_num = args.cluster_num  # number of clusters for Gaussian distribution
    error = args.error  # convergence condition for EM algorithm

    # load csv file
    data = ex3.load_data(filename)

    # dimension of the data
    if type(data[0]) is np.ndarray:
        dim = len(data[0])
    else:
        dim = 1
        data = data[:, np.newaxis]  # (n, ) -> (n, 1)

    # plot csv data
    if dim == 1:
        ax = ex3.plot_scatter_diag(
            data,
            title=f"GMM: {filename[2:-4]}, k={cluster_num}",
            xlabel="$x$",
            ylabel="Probability density",
        )
    elif dim <= 3:
        ax = ex3.plot_scatter_diag(
            data,
            title=f"GMM: {filename[2:-4]}, k={cluster_num}",
            xlabel="$x1$",
            ylabel="$x2$",
        )

    # initialize gmm parameters
    old_means, old_cov_matrix, old_weights = initialize_gmm(dim, cluster_num)

    # calculate log-likelihood
    old_ll = get_log_likelihood(data, old_means, old_cov_matrix, old_weights)
    # print(f"{old_ll=}\n{old_means=}\n{old_cov_matrix=}\n{old_weights=}")  # value check

    times = 0  # iteration
    ll_data = np.array([0, old_ll])  # transition data of log likelihood

    while True:  # EM algorithm
        times += 1
        new_params = em_algo(data, old_means, old_cov_matrix, old_weights)
        new_ll = get_log_likelihood(data, new_params[0], new_params[1], new_params[2])

        ll_data = np.vstack((ll_data, [times, new_ll]))

        if new_ll - old_ll < 0:  # check for monotonically increase
            print(f"ERROR: new_ll is smaller than old_ll\n{new_ll=}, {old_ll=}")

        if np.abs(new_ll - old_ll) < error:  # convergence check
            break

        # update parameters
        old_ll = new_ll
        old_means = new_params[0]
        old_cov_matrix = new_params[1]
        old_weights = new_params[2]

    # # value check
    # print(
    #     f"{new_ll=}\nnew_means={new_params[0]}\nnew_cov_matrix={new_params[1]}\nnew_weights={new_params[2]}"
    # )
    # print(f"{times=} : {np.abs(new_ll - old_ll)=}")

    # prepare plot data
    plot_num = len(data)
    if dim == 1:  # x
        x_value = np.linspace(np.min(data), np.max(data), len(data))[:, np.newaxis]
        mix_gauss = np.sum(
            calc_mix_gauss(x_value, new_params[0], new_params[1], new_params[2]), axis=0
        )
        axis_range = None

    elif dim == 2:  # x-y
        xmin = np.min(data[:, 0])
        xmax = np.max(data[:, 0])
        ymin = np.min(data[:, 1])
        ymax = np.max(data[:, 1])

        x1_value = np.linspace(xmin, xmax, plot_num)
        x2_value = np.linspace(ymin, ymax, plot_num)
        xx, yy = np.meshgrid(x1_value, x2_value)  # (plot_num, plot_num)

        mix_gauss = np.zeros((plot_num, plot_num))

        for n in range(plot_num):
            tmp = np.vstack((xx[n], yy[n])).T  # (plot_num, dim)
            mix_gauss[n, :] = np.sum(
                calc_mix_gauss(tmp, new_params[0], new_params[1], new_params[2]),
                axis=0,
            )
        x_value = (xx, yy)
        axis_range = [int(xmin - 1), int(xmax + 1), int(ymin - 1), int(ymax + 1)]

    # plot GMM and means
    plot_gmm(mix_gauss, new_params[0], x_value, axis_range, ax)

    # plot log-likelihood
    plot_one_line(
        ll_data,
        title=f"Log likelihood: {filename[2:-4]}, k={cluster_num}",
        xlabel="Iteration",
        ylabel="Log likelihood",
    )


if __name__ == "__main__":
    main()
