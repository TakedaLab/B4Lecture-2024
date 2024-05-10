"""This program performs a principal component analysis."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import ex3


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="performs a principal component analysis."
    )

    # csv file name
    parser.add_argument("--input-file", type=str, required=True, help="input csv file")

    # criteria for dimensionality reduction
    parser.add_argument(
        "--ccr-per", type=float, default=90, help="cumulative contribution rate [%]"
    )

    return parser.parse_args()


def calc_eig(dataset: np.ndarray) -> np.ndarray:
    """Calculate eigenvalues and transformation matrix.

    Args:
        dataset (np.ndarray): csv data.

    Returns:
        eig_sorted, trans_matrix: descending eigenvalue, transformation matrix.
    """
    dim = len(dataset[0])  # dimension of the data
    num = len(dataset)  # number of the data

    # calculate mean of x and x^2
    mean = np.zeros(dataset.shape)
    mean_xx = np.zeros(dataset.shape)
    for i in range(dim):
        mean[:, i] += sum(dataset[:, i]) / num
        mean_xx[:, i] += sum(dataset[:, i] ** 2) / num

    # standardization = sqrt(mean(x^2) - mean(x)^2)
    sd = (mean_xx - (mean**2)) ** 0.5
    sd_data = (dataset - mean) / sd

    # calculate covariance matrix
    cov_matrix = np.zeros((dim, dim))
    # upper triangular matrix
    for i in range(dim):
        for j in range(i + 1, dim):
            cov_matrix[i, j] = sum((sd_data[:, i] * sd_data[:, j]))
    # symmetric matrix, σ^2 = 1
    cov_matrix = ((cov_matrix + cov_matrix.T) / num) + np.eye(dim, dtype=float)

    # calculate eigenvalue and eigenvector
    eig, eig_vec = np.linalg.eig(cov_matrix)

    # sort eigenvalues in descending order
    sort_index = eig.argsort()[::-1]
    eig_sorted = np.sort(eig)[::-1]

    # transformation matrix
    trans_matrix = eig_vec[:, sort_index]

    return eig_sorted, trans_matrix


def calc_contribution(eig_sorted: np.ndarray) -> np.ndarray:
    """Calculate contribution rates.

    Args:
        eig_sorted (np.ndarray): descending eigenvalue.

    Returns:
        cont, cum_cont: contribution rates, cumulative contribution rates.
    """
    dim = len(eig_sorted)  # dimension of the data

    cont = np.zeros(dim)  # contribution rates
    cum_cont = np.zeros(dim)  # cumulative contribution rates

    for i in range(dim):
        cont[i] = eig_sorted[i] / dim
        cum_cont[i] = cum_cont[i - 1] + cont[i]

    return cont, cum_cont


def plot_bases(trans_matrix: np.ndarray, ax, cont: np.ndarray):
    """Plot the bases on a scatter plot.

    Args:
        trans_matrix (np.ndarray): transformation matrix.
        ax (Axis): Axis object of the scatter plot.
        cont (np.ndarray): contribution rates.

    Returns:
        ax: Axis object of the graph.
    """
    dim = len(trans_matrix[0])  # dimension of the data after transformation
    colors = ("b", "g", "r")  # color of line (blue, green, red)

    if dim == 2:  # x-y
        for i in range(dim):
            eig_vec = trans_matrix[:, i]
            cr = cont[i]  # contribution rate
            ax.axline(
                [0, 0], eig_vec, label=f"Contribution rate: {cr:3.3f}", color=colors[i]
            )

    elif dim == 3:  # x-y-z
        for i in range(dim):
            eig_vec = trans_matrix[:, i]
            cr = cont[i]  # contribution rate

            t = np.array([-2.5, 2.5])  # range of line
            line_points = t[:, np.newaxis] * eig_vec

            ax.plot(
                line_points[:, 0],
                line_points[:, 1],
                line_points[:, 2],
                label=f"Contribution rate: {cr:3.3f}",
                color=colors[i],
            )

    ax.legend()
    # plt.savefig("h_miyaji\\figs\\result2-1.png")
    plt.show()
    return ax


def main():
    """Load csv file and perform a principal component analysis."""
    # get argument
    args = parse_args()
    filename = args.input_file  # csv file name
    ccr = args.ccr_per  # cumulative contribution rate

    # load csv file
    data = ex3.load_data(filename)

    # dimension of the data
    dim = len(data[0])

    # plot csv data (2d or 3d)
    if dim <= 3:
        ax = ex3.plot_scatter_diag(data, title=filename[2:-4])

    # calculate eigenvalue and translation matrix
    eig, trans_matrix = calc_eig(data)
    # calculate contribution rates
    cont, cum_cont = calc_contribution(eig)
    print(f"contribution rates ({filename}):")
    print(cont)

    # plot base lines (2d or 3d)
    if dim <= 3:
        plot_bases(trans_matrix, ax, cont)

    # dimensionality reduction (3d -> 2d)
    if dim == 3:
        trans_data = (trans_matrix.T @ data.T).T  # y_i = A.T @ x_i
        data_2d = trans_data[:, :-1]  # 3d -> 2d
        ex3.plot_scatter_diag(
            data_2d, "After Dimensionality Reduction", xlabel="PC1", ylabel="PC2"
        )
        # plt.savefig("h_miyaji\\figs\\result2-2.png")
        plt.show()

    # criteria for dimensionality reduction (over 4d)
    if dim > 3:
        print(f"\ncumulative contribution rates ({filename}):")
        print(cum_cont)
        tmp = cum_cont[cum_cont < ccr / 100]
        print(f"\ncumulative contribution rate: {ccr}")
        print(f"dimensionality reduction: {len(cum_cont)} dim -> {len(tmp) + 1} dim")


if __name__ == "__main__":
    main()
