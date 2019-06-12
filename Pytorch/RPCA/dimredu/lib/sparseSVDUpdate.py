#! /usr/bin/env python

import numpy as np
from scipy.sparse.linalg import LinearOperator
from dimredu.lib.randomized_svd import randomized_svd


def sparseSVDUpdate(X, U, E, VT):
    """Compute a fast SVD decomposition.

    The is computes the SVD update of a matrix formed from the sum
    of a sparse matrix :math:`X` and a low rank matrix represented as an
    SVD.

    .. math::

       Y &= X + U \Sigma V^T \\
         &= U_Y \Sigma_Y V_Y^T

    Args:
        X: A sparse matrix.

        U, E, VT:  The SVD of a low rank matrix.

    Returns:
        The SVD of the sum of the matrices, truncated to
        the same number of singular values as the original low rank
        matrix.
    """
    k = U.shape[1]
    # Make sure we have matrices where we expect them.
    U = np.matrix(U)
    VT = np.matrix(VT)

    assert len(E.shape) == 1,\
        'E wrong dimension len(E.shape) == %d not 1' % len(E.shape)
    assert E.shape[0] == k,\
        'E wrong shape E.shape[0] == %d not %d' % (E.shape[0], k)
    assert VT.shape[0] == k,\
        'VT wrong shape VT.shape[0] == %d not %d' % (VT.shape[0], k)

    # # This is a dense version that one can compare against for debugging
    # Y = X + U*np.diag(E)*VT
    # [oU,oE,oVT] = np.linalg.svd(Y,full_matrices=False)

    def matmat(v):
        # This is fast since X is sparse making X*v fast.
        # Also, U*(E*(VT*v)) is fast if the number of columns
        # of U and the number of rows of VT is small.  Note,
        # the parentheses matter as the order of operations
        # is important!
        return X * v + U * (np.diag(E) * (VT * v))

    def matvec(v):
        v = np.matrix(v).T
        output = matmat(v)
        return np.array(output)[:, 0]

    def rmatmat(v):
        # This is just the transpose of matvec and is
        # fast for the same reasons.
        return X.T * v + VT.T * (np.diag(E) * (U.T * v))

    def rmatvec(v):
        v = np.matrix(v).T
        output = rmatmat(v)
        return np.array(output)[:, 0]

    # We use the above to create LinearOperators.  The idea is to
    # do a "matrix free" SVD.
    Y = LinearOperator([U.shape[0], VT.shape[1]],
                       matvec=matvec, matmat=matmat)
    YT = LinearOperator([VT.shape[1], U.shape[0]],
                        matvec=rmatvec, matmat=rmatmat)
    # This is our "matrix free" version of the SVD, closely based upon
    # the randomized_svd function in sklearn.utils.extmath.
    oU, oE, oVT = randomized_svd(Y, YT, k)
    # The outside routines expect these to be matrices.
    oU = np.matrix(oU)
    oVT = np.matrix(oVT)

    return [oU[:, :k], oE[:k], oVT[:k, :]]


def test_sparseSVDUpdate():
    from scipy.sparse import rand
    M = np.random.random(size=[3, 2])
    U, E, VT = np.linalg.svd(M, full_matrices=False)

    X = rand(3, 2, 0.5)

    print((sparseSVDUpdate(X, U, E, VT)))
    print((np.linalg.svd(M + X)))


if __name__ == '__main__':
    test_sparseSVDUpdate()
