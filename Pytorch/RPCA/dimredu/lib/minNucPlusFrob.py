#! /usr/bin/env python

import numpy as np
from dimredu.lib.shrink import shrink
from dimredu.lib.sparseSVDUpdate import sparseSVDUpdate
from dimredu.lib.nuclearNorm import nuclearNorm


def minNucPlusFrob(X, U, E, VT, mu, truncateK=0,
                   debug=False, guess=None):
    """Compute a fast minimization of nuclear norm plus Frobenius norm.

    The is computes the minium of the following objective.

    .. math::

        \| L \|_* + \mu / 2 \| L - (X + U*E*V^T) \|_F^2

    Args:
        X: A sparse array.

        U, E, VT: The SVD of a low matrix.

        mu: The value of :math:`\mu`.

        truncateK: Ignore the first truncateK singular values to use
                   the truncated nuclear norm

        debug:  Run the algorithm in debugging mode, with additional
                output and slower run-time.

        guess: An initial guess for the minimization.  In this case
                the minimization is in closed form, so it is merely
                used for debugging to see is the value of the
                objective is reduced.

    Returns:
        The SVD of :math:`L` that achieves the minimum.
    """
    assert X.shape[0] == U.shape[0], 'First dim of L and U must match'
    assert X.shape[1] == VT.shape[1], 'Last dim of L and VT must match'

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        Um = np.matrix(guess['U'])
        Em = np.matrix(np.diag(guess['E']))
        VTm = np.matrix(guess['VT'])
        before = objective(Um * Em * VTm, X, U, E, VT, mu,
                           truncateK=truncateK)
    # DEBUG ############################
    ####################################

    [Unew, Enew, VTnew] = sparseSVDUpdate(X, U, E, VT)

    # Don't shrink the first truncatedK singular
    # values to implement the truncated nuclear norm
    # as in
    Enew[truncateK:] = shrink(1. / mu, Enew[truncateK:])

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        Um = np.matrix(Unew)
        Em = np.matrix(np.diag(Enew))
        VTm = np.matrix(VTnew)
        after = objective(Um * Em * VTm, X, U, E, VT, mu,
                          truncateK=truncateK)
        assert before / after + 1e-7 >= 1., 'minNucPludFrob went up!'
    # DEBUG ############################
    ####################################

    return [Unew, Enew, VTnew]


def objective(L, X, U, E, VT, mu, truncateK=0):
    return (nuclearNorm(L, truncateK) +
            (mu / 2.) * np.linalg.norm(L - (X + U * np.diag(E) * VT), 'fro')**2)


def test_minNucPlusFrob():
    np.random.seed(123)
    A = np.matrix(np.random.random(size=[5, 5]))

    [U, E, VT] = np.linalg.svd(A)

    X = np.matrix(np.random.random(size=[5, 5]))
    # Needs to be bigger than 0.
    mu = 1.0

    [Umin, Emin, VTmin] = minNucPlusFrob(X, U, E, VT, mu,
                                         debug=True, guess={'U': U,
                                                            'E': E,
                                                            'VT': VT})

    Lmin = Umin * np.diag(Emin) * VTmin

    print(Umin)
    print(Emin)
    print(VTmin)

    print(Lmin)
    LminObj = objective(Lmin, X, U, E, VT, mu)
    print(('Should be smallest', LminObj))

    for i in range(5):
        perturb = np.random.random(size=[5, 5]) * 1e-5
        pObj = objective(Lmin + perturb, X, U, E, VT, mu)
        print(pObj)
        assert LminObj < pObj


if __name__ == '__main__':
    test_minNucPlusFrob()
