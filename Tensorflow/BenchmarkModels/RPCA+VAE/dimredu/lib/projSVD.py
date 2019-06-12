import numpy as np
import scipy.sparse as sp
from numba import jit


def projSVD(U, E, VT, u, v, returnVec=False):
    """A projection of an SVD onto a index set.
    We project an SVD on a (perhaps sparse) set of indices.

    Args:
        U,E,VT: The SVD of a matrix.

        u,v: The indices at which to evaluate the SVD.

        returnVec:  If True return a vector, otherwise return a
          sparse matrix.

    Returns:
       A sparse representation (either matrix or vector) with
       the evalutions of the matrix.  Note, the undefined entries are
       not necessarily 0!  They are merely unobserved.

    """
    # and we compute the projection of the new L onto Omega
    # Note, making the temp array and then creating the sparse
    # matrix all at once is *much* faster.

    assert U.shape[1] == len(E), 'shape mismatch'
    assert VT.shape[0] == len(E), 'shape mismatch'
    assert len(U.shape) == 2, 'U needs to be a matrix'
    assert len(VT.shape) == 2, 'VT need to be a matrix'
    assert len(E.shape) == 1, 'E need to be an array'
    m = U.shape[0]
    n = VT.shape[1]
    k = len(E)
    l = len(u)
    # if l is too close to m*n, just multiplying the matrices is faster
    if l == m * n:
        if returnVec:
            assert False, 'returnVec not set up for dense problems'
        else:
            # FIXME: Fix this correctly for the case where
            # this matrix has zeros
            return sp.csc_matrix(U * np.diag(E) * VT+1e-6)
    else:
        tmp = np.array(np.zeros([l]))
        _worker(U, E, VT, tmp, u, v, l, k)
        if returnVec:
            return tmp
        else:
            return sp.csc_matrix(sp.coo_matrix((tmp, (u, v)), shape=[m, n]))


@jit(nopython=True, cache=True)
def _worker(U, Sigma, VT, tmp, u, v, l, k):
    for i in range(l):
        tmp[i] = 0
        for j in range(k):
            tmp[i] += U[u[i], j] * Sigma[j] * VT[j, v[i]]


def test_projSVD():
    m = 4
    n = 3
    A = np.random.random(size=[m, n])
    U, E, VT = np.linalg.svd(A, full_matrices=False)
    Omega = np.random.random([m, n])
    Omega[Omega < 0.7] = 0
    Omega[Omega >= 0.7] = 1

    u = []
    v = []
    for i in range(m):
        for j in range(n):
            if Omega[i, j] == 1:
                u.append(i)
                v.append(j)

    O = projSVD(U, E, VT, u, v)
    print('O\n', O)
    print('(A * Omega)\n', (A * Omega))
    print('(O - A * Omega)\n', (O - A * Omega))


if __name__ == '__main__':
    test_projSVD()
