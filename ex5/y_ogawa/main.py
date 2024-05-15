"""GMMを用いたデータフィッティングを行う."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import ex4


def parse_args():
    """引数の取得を行う.

    filename : 読み込むファイル名
    """
    parser = argparse.ArgumentParser(description="主成分分析を行う")
    parser.add_argument("--filename", type=str, required=True, help="name of file")
    return parser.parse_args()


def main():
    """読み込んだデータでGMMを用いたデータフィッティング."""
    # 引数を受け取る
    args = parse_args()
    data = ex4.load_csv(args.filename)
    file_title = args.filename.replace(".csv", "")
    # print(data)
    dim = data.ndim
    ex4.make_scatter(data, dim, file_title + "_plot")


if __name__ == "__main__":
    main()
