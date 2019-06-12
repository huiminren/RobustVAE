#! /usr/bin/env python

import numpy as np
from dimredu.lib.shrink import shrink
from numba import jit


def minNDRSD(A, Yt, Yb, E, mu, hasWeave=True, debug=False, guess=None):
    """Compute a fast minimization of shrinkage plus Frobenius norm.

    The is computes the minium of the following objective.

    .. math::

       \arg \min_{S_D} \frac{\mu}{2} \Big \| \frac{1}{\mu} Y_t +
       \mathcal{S}_{\boldsymbol \epsilon}( P_{\Omega} ( S_D ) ) \Big \|_F^2 + \\
       \frac{\mu}{2} \Big \| \frac{1}{\mu} Y_t + P_{\Omega}(M_D) -
       (P_{\Omega}(D(L_G)) + P_{\Omega}(S_D)) \Big \|_F^2

    Args:
        A: A numpy array.

        Yt, Yb: Numpy arrays of Lagrange multipliers.

        E: A numpy array of error bounds.

        mu: The value of :math:`\mu`.

    Returns:
        The value of :math:`S` that achieves the minimum.

    """
    assert len(A.shape) == 1, 'A can only be a vector'
    assert A.shape == E.shape, 'A and E have  to have the same size'
    assert A.shape == Yt.shape, 'A and Yt have  to have the same size'
    assert A.shape == Yb.shape, 'A and Yb have  to have the same size'
    # Note, while the derivative is always zero when you use the
    # formula below, it is only a minimum if the second derivative is
    # positive.  The second derivative happens positive if and only
    # \mu is positive.
    assert mu >= 0., 'mu must be >= 0'

    mu = float(mu)

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        before = objective(guess, A, Yt, Yb, E, mu)
    # DEBUG ############################
    ####################################

    S = np.zeros(A.shape)
    _worker(A, Yt, E, Yb, mu, S)

    ####################################
    # DEBUG ############################
    if debug and (guess is not None):
        after = objective(S, A, Yt, Yb, E, mu)
        assert before / after + 1e-7 >= 1., 'minNDRSD went up!'
    # DEBUG ############################
    ####################################
    return S


@jit(nopython=True, cache=True)
def _worker(A, Yt, E, Yb, mu, S):
    for i in range(len(A)):
        if (1. / (2. * mu)) * (Yt[i] - mu * E[i] + Yb[i] + mu * A[i]) < -E[i]:
            S[i] = (1. / (2. * mu)) * \
                (Yt[i] - mu * E[i] + Yb[i] + mu * A[i])
        elif ((-E[i] < (1. / mu) * (Yb[i] + mu * A[i])) and
              ((1 / mu) * (Yb[i] + mu * A[i]) < E[i])):
            S[i] = (1. / mu) * (Yb[i] + mu * A[i])
        elif E[i] < (1. / (2. * mu)) * (-Yt[i] + mu * E[i] + Yb[i] + mu * A[i]):
            S[i] = (1. / (2. * mu)) * (-Yt[i] +
                                       mu * E[i] + Yb[i] + mu * A[i])
        else:
            term1 = (1. / mu) * Yt[i]
            term2 = (1. / mu) * Yb[i] + A[i] - E[i]
            term3 = (1. / mu) * Yb[i] + A[i] + E[i]
            Sp = (mu / 2.) * (term1 * term1 + term2 * term2)
            Sm = (mu / 2.) * (term1 * term1 + term3 * term3)
            if Sp < Sm:
                S[i] = E[i]
            else:
                S[i] = -E[i]


def objective(S, A, Yt, Yb, E, mu):
    temp1 = (mu / 2.) * np.linalg.norm((1. / mu)
                                       * Yt + np.abs(shrink(E, S)))**2
    temp2 = (mu / 2.) * np.linalg.norm((1. / mu) * Yb + A - S)**2
    return temp1 + temp2


def data(which):
    # Test data sets including those that given problems before.
    if which == 'problem1':
        SOrig = np.array([-0.00000e+00,  -6.44347e-02,   1.46198e-01,   4.89326e+00,
                          4.86620e+00,  -6.44347e-02,  -5.20417e-18,  -1.38359e-02,
                          3.11687e-01,   5.01579e+00,   1.46198e-01,  -1.38359e-02,
                          -0.00000e+00,  -7.29722e-02,   4.96465e+00,   4.89326e+00,
                          3.11687e-01,  -7.29722e-02,  -5.20417e-18,   4.20675e-02,
                          4.86620e+00,   5.01579e+00,   4.96465e+00,   4.20675e-02,
                          -0.00000e+00])
        A = np.array([1.38778e-17,  -1.36229e-01,  -1.51878e-01,   4.80873e+00,
                      4.81636e+00,  -1.36229e-01,  -2.77556e-17,  -1.46025e-01,
                      1.02459e-01,   5.02334e+00,  -1.51878e-01,  -1.46025e-01,
                      -1.38778e-17,  -2.02653e-01,   4.93558e+00,   4.80873e+00,
                      1.02459e-01,  -2.02653e-01,  -1.38778e-17,  -6.61490e-02,
                      4.81636e+00,   5.02334e+00,   4.93558e+00,  -6.61490e-02,
                      -6.93889e-18])
        Yt = np.array([0.00000e+00,   3.96973e-02,   1.39967e+00,   0.00000e+00,
                       0.00000e+00,   3.96973e-02,   4.99255e-18,   3.85058e-01,
                       1.52151e+00,   6.40735e-02,   1.39967e+00,   3.85058e-01,
                       2.49628e-18,   3.01137e-01,   8.15302e-03,   0.00000e+00,
                       1.52151e+00,   3.01137e-01,  -1.49777e-17,   7.58762e-01,
                       0.00000e+00,   6.40735e-02,   8.15302e-03,   7.58762e-01,
                       0.00000e+00])
        Yb = np.array([0.00000e+00,   3.96973e-02,   1.39967e+00,   0.00000e+00,
                       0.00000e+00,   3.96973e-02,   4.99255e-18,   3.85058e-01,
                       1.20474e+00,   6.40735e-02,   1.39967e+00,   3.85058e-01,
                       2.49628e-18,   3.01137e-01,   0.00000e+00,   0.00000e+00,
                       1.20474e+00,   3.01137e-01,  -1.49777e-17,   6.40717e-01,
                       0.00000e+00,   6.40735e-02,   0.00000e+00,   6.40717e-01,
                       0.00000e+00])
        E = np.array([0.00000e+00,   4.32591e-03,   9.95856e-02,   5.00000e+00,
                      5.00000e+00,   4.32591e-03,   0.00000e+00,   7.57776e-03,
                      3.11687e-01,   5.00000e+00,   9.95856e-02,   7.57776e-03,
                      0.00000e+00,   1.73463e-02,   5.00000e+00,   5.00000e+00,
                      3.11687e-01,   1.73463e-02,   0.00000e+00,   4.20675e-02,
                      5.00000e+00,   5.00000e+00,   5.00000e+00,   4.20675e-02,
                      0.00000e+00])
        mu = 2.8780102048
        return A, Yt, Yb, E, mu

    if which == 'problem2':
        A = np.array([0.,  0.15854,  0.71644,  5.,  5.,  0.15854,
                      0.,  0.20875,  1.19954,  5.,  0.71644,  0.20875,
                      0.,  0.31242,  5.,  5.,  1.19954,  0.31242,
                      0.,  0.47801,  5.,  5.,  5.,  0.47801,  0.])
        Yt = np.array([0.,  0.00741,  0.02558,  0.,  0.,  0.00741,
                       0.,  0.00957,  0.02849,  0.,  0.02558,  0.00957,
                       0.,  0.01373,  0.,  0.,  0.02849,  0.01373,
                       0.,  0.01947,  0.,  0.,  0.,  0.01947,  0.])
        Yb = np.array([0.,  0.00784,  0.03542,  0.,  0.,  0.00784,
                       0.,  0.01032,  0.05931,  0.,  0.03542,  0.01032,
                       0.,  0.01545,  0.,  0.,  0.05931,  0.01545,
                       0.,  0.02363,  0.,  0.,  0.,  0.02363,  0.])
        E = np.array([0.00000e+00,   4.32591e-03,   9.95856e-02,   5.00000e+00,
                      5.00000e+00,   4.32591e-03,   0.00000e+00,   7.57776e-03,
                      3.11687e-01,   5.00000e+00,   9.95856e-02,   7.57776e-03,
                      0.00000e+00,   1.73463e-02,   5.00000e+00,   5.00000e+00,
                      3.11687e-01,   1.73463e-02,   0.00000e+00,   4.20675e-02,
                      5.00000e+00,   5.00000e+00,   5.00000e+00,   4.20675e-02,
                      0.00000e+00])
        mu = 0.304172303889
        return A, Yt, Yb, E, mu

    if which == 'randomNormal':
        np.random.seed(1234)
        A = np.random.normal(size=[5])
        Yt = np.random.normal(size=A.shape)
        Yb = np.random.normal(size=A.shape)
        E = np.ones(A.shape) * 1e-1
        mu = 0.1
        return A, Yt, Yb, E, mu

    if which == 'simple':
        A = np.array([1])
        Yt = np.array([2])
        Yb = np.array([3])
        E = np.array([4])
        mu = np.array([5])
        return A, Yt, Yb, E, mu


data.sets = ['problem1', 'problem2', 'randomNormal', 'simple']


def plot_objective():
    A, Yt, Yb, E, mu = data('randomNormal')
    print()
    print('A, Yt, Yb, E, mu')
    print(A, Yt, Yb, E, mu)
    Smin = minNDRSD(A, Yt, Yb, E, mu, hasWeave=True)
    print('Smin')
    print(Smin)
    SminObj = objective(Smin, A, Yt, Yb, E, mu)
    print('Should be smallest', SminObj)

    print('random')
    for i in range(5):
        perturb = np.random.normal(size=A.shape) * 1e-2
        print(perturb)
        pObj = objective(Smin + perturb, A, Yt, Yb, E, mu)
        print(pObj)
        assert SminObj <= pObj

    X = []
    Y = []
    print('linear')
    perturb = np.random.normal(A.shape)
    for s in np.linspace(-1e+1, +1e+1, 100):
        pObj = objective(Smin + perturb * s, A, Yt, Yb, E, mu)
        X.append(s)
        Y.append(pObj)

    import matplotlib.pylab as py
    py.figure(1)
    py.plot(X, Y)
    py.show()


def test_minNDRSD():
    A, Yt, Yb, E, mu = data('randomNormal')

    print()
    print('A, Yt, Yb, E, mu')
    print(A, Yt, Yb, E, mu)
    Smin = minNDRSD(A, Yt, Yb, E, mu, hasWeave=True)
    print('Smin')
    print(Smin)
    SminObj = objective(Smin, A, Yt, Yb, E, mu)
    print('Should be smallest', SminObj)

    for i in range(10):
        # This should be smaller that E, otherwise the objective
        # is flat.
        perturb = np.random.normal(size=A.shape) * 1e-3
        pObj = objective(Smin + perturb, A, Yt, Yb, E, mu)
        print(pObj)
        assert SminObj <= pObj


def test_minNDRSD2():
    for hasWeave in [True, False]:
        for which in data.sets:
            A, Yt, Yb, E, mu = data(which)
            print()
            print('A, Yt, Yb, E, mu')
            print(A, Yt, Yb, E, mu)
            Smin = minNDRSD(A, Yt, Yb, E, mu, hasWeave=hasWeave,
                            debug=True, guess=np.random.normal(size=A.shape))
            print('Smin')
            print(Smin)
            SminObj = objective(Smin, A, Yt, Yb, E, mu)
            print('Should be smallest', SminObj)

            for i in range(10):
                # This should be smaller that E, otherwise the objective
                # is flat.
                perturb = np.random.normal(size=A.shape) * 1e-3
                pObj = objective(Smin + perturb, A, Yt, Yb, E, mu)
                print(pObj)
                assert SminObj <= pObj


if __name__ == '__main__':
    test_minNDRSD()
    test_minNDRSD2()
    # plot_objective()
