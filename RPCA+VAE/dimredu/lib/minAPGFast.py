#! /usr/bin/env python

import numpy as np
from dimredu.lib.shrink import shrink
from dimredu.lib.sparseSVDUpdate import sparseSVDUpdate
from dimredu.lib.nuclearNorm import nuclearNorm


def minAPGFast(m, n, A, AT, b, mu, guess, truncateK=0,
               tau=10.0, debug=False, maxIter=1):
    """Compute a fast *single interation* of the minimization using the
    APG algorithm from:

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

        guess: An initial guess for the minimization.  It is also
               used for debugging to make sure the value of the
               objective is smaller.

        truncateK: Ignore the first truncateK singular values to use
                   the truncated nuclear norm

        tau: The Lipschitz constant for the problem of interest. Need to
             figure out numerically unless known analytically.

        maxIter: The number of iterations to run the solver.  One is
             sometimes enough.

        debug:  Run the algorithm in debugging mode, with additional
                output and slower run-time.

    Returns:
        The SVD of :math:`L` that achieves the minimum.

    """
    assert len(b.shape) == 1, 'b must be a vector'

    # The SVD of the L we guess
    U = np.matrix(guess['U'])
    E = guess['E']
    VT = np.matrix(guess['VT'])
    # The indicies in Omega at which L is observed
    u = guess['u']
    v = guess['v']

    assert len(u) == len(v) == len(b), 'length of u, v, and b must be the same'

    # To make it consistent with the APG paper we make
    # the following transformation.
    muP = 1. / mu

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        Um = np.matrix(U)
        Em = np.matrix(np.diag(E))
        VTm = np.matrix(VT)
        before = objective(Um * Em * VTm, A, U, E, VT, u, v, b, mu,
                           truncateK=truncateK)
        print('minAPGFast before', before)
    # DEBUG ############################
    ####################################

    # Note, this is not the same algorithm as in minAPG.py.
    # This algorithm does not use the same normalization constants!
    # In fact, this algorithm does not have the support of theory
    # as in minAPG.py!
    # This code requires a rank one update to be the same as
    # minAPG.py.
    for i in range(maxIter):
        # Note, this is sparse since the AT will return a
        # sparse matrix!
        tmpOmega = -(1. / tau) * AT(A(U, E, VT, u, v) - b, u, v, m, n)
        # So, this will be a sparseSVDUpdate of
        # Y - (1./tau)*AT(A(Y)-b)
        # which, in the notation here, is
        # U*E*VT + tmpOmega
        # and we rearrange terms to get what we expect
        # for sparseSVDUpdate
        # sparseSVDUpdate(tmpOmega, U, E, VT)

        # NOTE:  tmpOmega is a sparse matrix!  This is ok,
        # since we only use it for sparseSVDUpdate, which expects
        # that
        Unew, Enew, VTnew = sparseSVDUpdate(tmpOmega, U, E, VT)
        Enew[truncateK:] = shrink(muP / tau, Enew[truncateK:])
        U = Unew
        E = Enew
        VT = VTnew

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        Um = np.matrix(Unew)
        Em = np.matrix(np.diag(Enew))
        VTm = np.matrix(VTnew)
        after = objective(Um * Em * VTm, A, Unew, Enew, VTnew,
                          u, v, b, mu,
                          truncateK=truncateK,
                          debug=True)
        print('minAPGFast after', after)

        # Compute the slow solution
        print('Using slow solver to help debug.  This can be very slow on large problems.')
        from dimredu.lib.minAPG import minAPG as minAPGSlow
        from dimredu.lib.EDM import K, KAdjoint

        def ASlow(X, u=u, v=v):
            n = len(u)
            output = np.zeros([n])
            Tmp = np.array(K(X))
            for i in range(n):
                output[i] = Tmp[u[i], v[i]]
            return output

        def ATSlow(x, u=u, v=v, m=m, n=n):
            Tmp = np.matrix(np.zeros([m, n]))
            for i in range(len(u)):
                Tmp[u[i], v[i]] = x[i]
            return KAdjoint(Tmp, symmetric=False)
        [Uexact, Eexact, VTexact] = minAPGSlow(m, n,
                                               ASlow,
                                               ATSlow,
                                               b, mu,
                                               maxIter=maxIter,
                                               tau=tau,
                                               guess=guess)
        Umexact = np.matrix(Uexact)
        Emexact = np.matrix(np.diag(Eexact))
        VTmexact = np.matrix(VTexact)
        slowAfter = objective(Umexact * Emexact * VTmexact,
                              A, Uexact, Eexact, VTexact,
                              u, v, b, mu,
                              truncateK=truncateK,
                              debug=True)
        print('minAPGSlow after', slowAfter)
        ###

        for i in range(5):
            perturb = np.random.random(size=[m, n]) * 1e-5
            Ltmp = Um * Em * VTm + perturb
            Utmp, Etmp, VTtmp = np.linalg.svd(Ltmp)
            pObj = objective(Ltmp, A, Utmp, Etmp, VTtmp,
                             u, v, b, mu,
                             truncateK=truncateK)
            print(pObj, end=' ')
            if after < pObj:
                print('bigger :-)')
            else:
                print('smaller :-(')
                assert False, 'Only local minimum'
        assert before / after + 1e-7 >= 1., 'minAPGFast went up!'
    # DEBUG ############################
    ####################################

    return [Unew, Enew, VTnew]


def objective(L, A, U, E, VT, u, v, b, mu, truncateK=0, debug=False):
    if debug:
        print('minAPGFast objective term1', nuclearNorm(L, truncateK))
        print('minAPGFast objective term2', \
            (mu / 2.) * np.linalg.norm(A(U, E, VT, u, v) - b)**2)
        # print 'minAPGFast objective b', b
        # print 'minAPGFast objective A', A(U, E, VT, u, v)
    return (nuclearNorm(L, truncateK) +
            (mu / 2.) * np.linalg.norm(A(U, E, VT, u, v) - b)**2)


def test_minAPGFast():
    from dimredu.lib.EDM import K, KAdjoint, KFast, KAdjointFast
    from dimredu.lib.minAPG import minAPG
    from dimredu.lib.minAPG import objective as objectiveSlow

    np.random.seed(1234)
    size = 2
    m = size
    n = size

    X = np.matrix(np.random.random([m, n]))
    guess = {}
    guess['U'], guess['E'], guess['VT'] = np.linalg.svd(X, full_matrices=False)
    u, v = np.meshgrid(list(range(m)), list(range(n)), indexing='ij')
    guess['u'] = u.flatten()
    guess['v'] = v.flatten()

    def A(X):
        return np.reshape(np.array(K(X)), [m * n])

    def AT(x):
        X = np.reshape(x, [m, n])
        return KAdjoint(X, symmetric=False)

    def AFast(U, E, VT, u, v):
        return KFast(U, E, VT, u, v)

    def ATFast(x, u, v, m, n):
        return KAdjointFast(x, u, v, m, n)

    b = np.random.random(size=[m * n])

    # Needs to be bigger than 0.
    mu = 1.0

    # When maxIter=1 these should be identical
    maxIter = 10
    [Umin1, Emin1, VTmin1] = minAPG(m, n, A, AT, b, mu,
                                    guess=guess,
                                    maxIter=maxIter)
    [Umin2, Emin2, VTmin2] = minAPGFast(m, n, AFast, ATFast,
                                        b.flatten(), mu,
                                        guess=guess,
                                        maxIter=maxIter,
                                        debug=False)

    Lmin1 = Umin1 * np.diag(Emin1) * VTmin1
    Lmin2 = Umin2 * np.diag(Emin2) * VTmin2

    print()
    print('Lmin1')
    print(Lmin1)
    print('Lmin2')
    print(Lmin2)
    LminObj1 = objectiveSlow(Lmin1, A, b, mu)
    print('Should be the same', LminObj1)
    LminObj2 = objective(Lmin2, AFast, Umin2, Emin2, VTmin2,
                         guess['u'], guess['v'], b, mu)
    print('Should be the same', LminObj2)

    # assert np.abs(LminObj1 - LminObj2) < 1e-7, 'The two solutions do not match'


if __name__ == '__main__':
    test_minAPGFast()
