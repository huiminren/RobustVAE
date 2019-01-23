import numpy as np
from numba import jit
from dimredu.lib.shrink import shrink


def minShrink1Plus2Norm(A, E, lam, mu):
    """Compute a fast minimization of shrinkage plus Frobenius norm.

    The is computes the minium of the following objective.

    .. math::

        \lambda \| \mathcal{S}_{\epsilon}( S_{ij} ) \|_1 +
        \mu / 2 \| S_{ij} - A_{ij} \|_F^2

    Args:
        A: A numpy array.

        E: A numpy array of error bounds.

        lam: The value of :math:`\lambda`.

        mu: The value of :math:`\mu`.

    Returns:
        The value of :math:`S` that achieves the minimum.
    """
    assert len(A.shape) == 1, 'A can only be a vector'
    assert A.shape == E.shape, 'A and E have  to have the same size'
    # Note, while the derivative is always zero when you use the
    # formula below, it is only a minimum if the second derivative is
    # positive.  The second derivative happens to be \mu.
    assert mu >= 0., 'mu must be >= 0'

    S = np.zeros(A.shape)
    _worker(A, E, lam, mu, S)
    return S


@jit(nopython=True, cache=True)
def _worker(A, E, lam, mu, S):
    for i in range(len(A)):
        if (lam / mu + A[i]) < -E[i]:
            S[i] = lam / mu + A[i]
        elif -E[i] < A[i] < E[i]:
            S[i] = A[i]
        elif E[i] < (-lam / mu + A[i]):
            S[i] = -lam / mu + A[i]
        else:
            Sp = (mu / 2.) * (E[i] - A[i]) * (E[i] - A[i])
            Sm = (mu / 2.) * (-E[i] - A[i]) * (-E[i] - A[i])
            if Sp < Sm:
                S[i] = E[i]
            else:
                S[i] = -E[i]



def objective(S, A, E, lam, mu):
    return (lam * np.linalg.norm(shrink(E, S), 1) +
            (mu / 2.) * (np.linalg.norm(S - A)**2))


def test_minShrink1Plus2Norm():
    # FIXME: This is a maximum for some values.  This happens to be
    # one such value.
    np.random.seed(1234)
    A = np.random.random(size=[5])
    E = np.ones([5]) * 1e-4
    lam = np.random.uniform()
    # Needs to be bigger than 0.
    mu = 0.1

    print()
    print('A, E, lam, mu')
    print((A, E, lam, mu))
    Smin = minShrink1Plus2Norm(A, E, lam, mu)
    print('Smin')
    print(Smin)
    SminObj = objective(Smin, A, E, lam, mu)
    print(('Should be smallest', SminObj))

    for i in range(5):
        # This should be smaller that E, otherwise the objective
        # is flat.
        perturb = np.random.random(size=[5]) * 1e-3
        pObj = objective(Smin + perturb, A, E, lam, mu)
        print(pObj)
        assert SminObj <= pObj


if __name__ == '__main__':
    test_minShrink1Plus2Norm()
