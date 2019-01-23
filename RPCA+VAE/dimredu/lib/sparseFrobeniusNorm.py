#! /usr/bin/env python

import numpy as np


def sparseFrobeniusNorm(X):
    """ Compute a the Frobenius norm of the observed entries of a sparse matrix.

    Args:
        X: a sparse matrix

    Returns:
        The square root of the sum of the squares of the observed entries.
    """
    tmp = X.multiply(X)
    return np.sqrt(tmp.sum())


def test_sparseFrobeniusNorm():
    from scipy.sparse import rand

    X = rand(3, 4, 0.5)

    print((sparseFrobeniusNorm(X)))
    print((np.linalg.norm(X.todense(), 'fro')))


if __name__ == '__main__':
    test_sparseFrobeniusNorm()
