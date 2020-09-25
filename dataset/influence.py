import numpy as np


def distance_based_influence(p: np.ndarray, q: np.ndarray):
    d = p - q
    return 1 / (d[:, 0] ** 2 + d[:, 1] ** 2)


if __name__ == '__main__':
    pass
