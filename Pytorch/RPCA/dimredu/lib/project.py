#! /usr/bin/python


def Pi(Omega, X):
    """The projection operator.

    This implementation is intentionally slow but transparent as
    to the mathematics.

    Args:
        Omega: A matrix of elements that evaluate to True and False and we
               project onto the True elements (and False elements are 0 in
               the output matrix).

        X: The matrix to project.

    Returns:
        The projected matrix
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not Omega[i, j]:
                X[i, j] = 0
    return X


def test_Pi():
    import numpy as np
    D = np.random.random(size=[3, 3])
    Omega = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 0]])
    A = Pi(Omega, D)
    for i in range(3):
        for j in range(3):
            if Omega[i, j]:
                assert A[i, j] == D[i, j]
            else:
                assert A[i, j] == 0
