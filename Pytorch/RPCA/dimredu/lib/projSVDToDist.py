import numpy as np
import scipy.sparse as sp
from numba import jit


def projSVDToDist(U, E, VT, u, v, returnVec=False):
    """A projection of an SVD onto a index set with a conversion to
    a distance matrix.

    Args:
        U,E,VT: The SVD of a matrix.

        u,v: The indices at which to evaluate the SVD.

        returnVec:  If True return a vector, otherwise return a
          sparse matrix.

    Returns:
        A sparse matrix with the evalutions of the matrix and
        then converted to a distance matrix.  Note, the undefined
        entries are not necessarily 0!  They are merely unobserved.

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
    l = len(u)
    tmp = np.array(np.zeros([l]))
    Sigma = np.diag(E)
    _worker(U, Sigma, VT, tmp, u, v, l)
    if returnVec:
        return tmp
    else:
        return sp.csc_matrix(sp.coo_matrix((tmp, (u, v)),
                                           shape=[m, n]))


@jit(nopython=True, cache=True)
def _worker(U, Sigma, VT, tmp, u, v, l):
    for i in range(l):
        tmp[i] = ((U[u[i], :] * Sigma * VT[:, u[i]] +
                   U[v[i], :] * Sigma * VT[:, v[i]] -
                   2 * U[u[i], :] * Sigma * VT[:, v[i]]))[0, 0]


def test_projSVDToDist():
    np.random.seed(1234)
    m = 4
    n = 4
    A = np.matrix(np.random.random(size=[m, n]))
    G = A * A.T
    U, E, VT = np.linalg.svd(G, full_matrices=False)
    Omega = np.random.random([m, n])
    Omega[Omega < 0.7] = 0
    Omega[Omega >= 0.7] = 1

    my1 = np.matrix(np.ones([m, 1]))
    dG = np.matrix(np.diag(G)).T
    D = dG * my1.T + my1 * dG.T - 2 * G

    u = []
    v = []
    for i in range(m):
        for j in range(n):
            if Omega[i, j] == 1:
                u.append(i)
                v.append(j)

    O = projSVDToDist(U, E, VT, u, v)
    print(O)
    print((np.multiply(D, Omega)))
    print((O - np.multiply(D, Omega)))

    from dimredu.lib.EDM import KFast
    OK = KFast(U, E, VT, u, v)
    print(u)
    print(v)
    print(OK)


if __name__ == '__main__':
    test_projSVDToDist()
