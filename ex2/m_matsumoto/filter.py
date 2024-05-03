"""Filter Funcs."""

from typing import Callable

import numpy as np


def sinc(x: np.ndarray) -> np.ndarray:
    """Sinc Function.

    Args:
        x (np.ndarray): input

    Returns:
        np.ndarray: output
    """
    return np.piecewise(x, [x == 0, x != 0], [1, lambda x: np.sin(x) / x])


def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Convolute Them.

    Args:
        left (np.ndarray): input0
        right (np.ndarray): input1

    Returns:
        np.ndarray: Convoluted output.
    """
    ret: np.ndarray = np.zeros(len(left) + len(right) - 1)

    for i in range(len(left)):
        ret[i : i + len(right)] += left[i] * right

    return ret


def bef(
    left: float,
    right: float,
    tap: int,
    rate: int,
    window: Callable[[int], np.ndarray] = np.hamming,
) -> np.ndarray:
    """Band Elimination Filter.

    Args:
        left (float): cut-off frequency (left<right)
        right (float): cut-off frequency (left<right)
        tap (int): the number of taps on filter (次数)
        rate (int): frequency rate
        window (function, optional): window function

    Returns:
        np.ndarray: Band Elimination Filter
    """
    if left > right:
        raise ValueError(f"frequency must be {left}<{right}")

    # Even Numberize.
    tap = tap + 1 if tap % 2 != 0 else tap

    # Convert to Angular Frequency.
    w_left: float = 2 * np.pi * left / rate
    w_right: float = 2 * np.pi * right / rate

    # Digital Pulse.
    n: np.ndarray = np.arange(-tap // 2, (tap // 2) + 1)

    return (
        sinc(np.pi * n)
        + (w_left / np.pi) * sinc(w_left * n)
        - (w_right / np.pi) * sinc(w_right * n)
    ) * window(tap + 1)


if __name__ == "__main__":
    print("asdf")
