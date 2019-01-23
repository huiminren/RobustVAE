import numpy as np
from numba import jit
from dimredu.lib.shrink import shrink


def minShrink2Plus2Norm(C, D, E, mu):
    """Compute a fast minimization of shrinkage with a Frobenius norm plus
    a Frobenius norm.

    The is computes the minium of the following objective with
    respect to B

    .. math::

        \frac{\mu}{2} \| C_{ij} + \mathcal{S}_{E_{ij}}( B_{ij} ) \|_F^2 +
        \frac{\mu}{2} \| D_{ij} - B_{ij} ) \|_F^2 +

    Args:
        C: A numpy array.

        D: A numpy array.

        mu: The value of :math:`\mu`.

    Returns:
        The value of :math:`S` that achieves the minimum.
    """
    assert len(C.shape) == 1, 'A can only be a vector'
    assert len(D.shape) == 1, 'A can only be a vector'
    assert len(E.shape) == 1, 'A can only be a vector'
    assert C.shape == D.shape == E.shape, 'C, D, E have to have the same size'
    # Note, while the derivative is always zero when you use the
    # formula below, it is only a minimum if the second derivative is
    # positive.  The second derivative happens to be \mu.
    assert mu >= 0., 'mu must be >= 0'

    B = np.zeros(C.shape)
    _worker(C, D, E, mu, B)
    return B


@jit(nopython=True, cache=True)
def _worker(C, D, E, mu, B):
    for i in range(len(C)):
        if -E[i] < D[i] < E[i]:
            B[i] = D[i]
        else:
            Bp = (mu / 2.) * (C[i])**2 + (mu / 2.) * (D[i] - E[i])**2
            Bm = (mu / 2.) * (C[i])**2 + (mu / 2.) * (D[i] + E[i])**2
            if Bp < Bm:
                B[i] = E[i]
            else:
                B[i] = -E[i]


def objective(C, D, E, mu, B):
    return ((mu / 2.) * np.linalg.norm(C + shrink(E, B))**2 +
            (mu / 2.) * np.linalg.norm(D + B)**2)


def test_minShrink2Plus2Norm():
    # np.random.seed(1234)
    C = np.random.random(size=[5])
    D = np.random.random(size=[5])
    # Needs to be bigger than 0.
    E = np.random.uniform(size=[5])
    mu = np.random.uniform()

    print()
    print('C, D, E, mu')
    print(C, D, E, mu)
    Bmin = minShrink2Plus2Norm(C, D, E, mu)
    print('Bmin')
    print(Bmin)
    BminObj = objective(C, D, E, mu, Bmin)
    print(('Should be smallest', BminObj))

    for i in range(5):
        # This should be smaller that E, otherwise the objective
        # is flat.
        perturb = np.random.random(size=[5]) * 1e-3
        pObj = objective(C, D, E, mu, Bmin+perturb)
        print(pObj, pObj - BminObj)
        assert BminObj <= pObj


if __name__ == '__main__':
    test_minShrink2Plus2Norm()
