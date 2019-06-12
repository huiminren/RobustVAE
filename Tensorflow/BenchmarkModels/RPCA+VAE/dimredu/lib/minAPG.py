#! /usr/bin/env python

import numpy as np
from dimredu.lib.shrink import shrink
from dimredu.lib.nuclearNorm import nuclearNorm


def minAPG(m, n, A, AT, b, mu, maxDim=None, truncateK=0,
           tau=10.0, maxIter=1,
           debug=False, guess=None):
    """Compute a minimization using the APG algorithm from:

    An accelerated proximal gradient algorithm for nuclear
    norm regularized linear least squares problems
    Kim-Chuan Toh and Sangwoon Yun

    The is computes the minium of the following objective.

    .. math::

        \| L \|_* + \mu / 2 \| A(L) - b \|_2^2

    Note, the :math:`\mu` here is :math:`\frac{1}{\mu}` in
    the above paper.

    Args:
        m, n:  The dimension of the output.

        A: A linear mapping from a matrix to a vector

        AT: The adjoint mapping of A

        b: A vector in the proximal function

        mu: The value of :math:`\mu`.

        maxDim: The maximum rank of the returned answer

        truncateK: Ignore the first truncateK singular values to use
                   the truncated nuclear norm

        tau: The Lipschitz constant for the problem of interest. Need to
             figure out numerically unless known analytically.

        maxIter: The number of iterations to run the solver.  One is
             sometimes enough.

        debug:  Run the algorithm in debugging mode, with additional
                output and slower run-time.

        guess: An initial guess for the minimization.  It is also
               used for debugging to make sure the value of the
               objective is smaller.

    Returns:
        The SVD of :math:`L` that achieves the minimum.
    """
    assert len(b.shape) == 1, 'b must be a vector'
    if debug:
        tmpX = np.zeros([m, n])
        tmpb = A(tmpX)
        assert len(tmpb.shape) == 1, 'output of A must be a vector'
        assert b.shape[0] == tmpb.shape[0], \
            'the size of b and the output of A must be the same'
        try:
            AT(tmpb)
        except:
            assert False, 'Must be able to call AT with the output of A'
        try:
            A(AT(tmpb))
        except:
            assert False, 'Must be able to call A with the output of AT'

    # To make it consistent with the APG paper we make
    # the following transformation.
    muP = 1. / mu

    if guess is not None:
        Um = np.matrix(guess['U'])
        Em = np.matrix(np.diag(guess['E']))
        VTm = np.matrix(guess['VT'])
        X0 = Um * Em * VTm
        X1 = X0.copy()
    else:
        X0 = np.zeros([m, n])
        X1 = np.zeros([m, n])

    ####################################
    # DEBUG ############################
    if debug:
        before = objective(Um * Em * VTm, A, b, mu,
                           truncateK=truncateK)
    # DEBUG ############################
    ####################################

    t0 = 1.
    t1 = 1.
    for k in range(maxIter):
        Y = X1 + ((t0 - 1.) / t1) * (X1 - X0)
        G = Y - (1. / tau) * AT(A(Y) - b)
        U, E, VT = np.linalg.svd(G)
        E[truncateK:] = shrink(muP / tau, E[truncateK:])
        X0 = X1
        X1 = U * np.diag(E) * VT
        t0 = t1
        t1 = (1. + np.sqrt(1. + 4. * t1 * t1)) / 2.

    Unew, Enew, VTnew = np.linalg.svd(X1)

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        Um = np.matrix(Unew)
        Em = np.matrix(np.diag(Enew))
        VTm = np.matrix(VTnew)
        after = objective(Um * Em * VTm, A, b, mu,
                          truncateK=truncateK)
        assert before / after + 1e-7 >= 1., 'minAPG went up!'
    # DEBUG ############################
    ####################################

    return [Unew, Enew, VTnew]


def objective(L, A, b, mu, truncateK=0):
    return (nuclearNorm(L, truncateK) +
            (mu / 2.) * np.linalg.norm(A(L) - b)**2)


def test_minAPG():
    from .EDM import K, KAdjoint
    np.random.seed(1234)
    size = 2
    m = size
    n = size

    def A(X):
        return np.reshape(K(X), [m * n])

    def AT(x):
        X = np.reshape(x, [m, n])
        return KAdjoint(X)

    b = np.random.random(size=[m * n])

    # Needs to be bigger than 0.
    mu = 1.0

    [Umin, Emin, VTmin] = minAPG(m, n, A, AT, b, mu)

    Lmin = Umin * np.diag(Emin) * VTmin

    print(Lmin)
    LminObj = objective(Lmin, A, b, mu)
    print('Should be smallest', LminObj)

    for i in range(5):
        perturb = np.random.random(size=[size, size]) * 1e-5
        pObj = objective(Lmin + perturb, A, b, mu)
        print(pObj, end=' ')
        if LminObj < pObj:
            print('bigger :-)')
        else:
            print('smaller :-(')
        assert LminObj < pObj


if __name__ == '__main__':
    test_minAPG()
