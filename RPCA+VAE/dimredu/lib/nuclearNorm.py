#! /usr/bin/env python

import numpy as np


def nuclearNorm(X, truncateK=0):
    """Compute the nuclear norm of a matrix.

    Args:
        X: the matrix

    Returns:
        The sum of the singular values
    """
    dummy, E, dummy = np.linalg.svd(X)
    return np.sum(E[truncateK:])


def test_nuclearNorm():
    X = np.random.random(size=[3, 5])

    print((nuclearNorm(X)))


if __name__ == '__main__':
    test_nuclearNorm()
