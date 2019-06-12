#! /usr/bin/env python
"""A matrix completion solver that implements Algorithm 6 (Matrix
Completion via Inexact ALM Method) from

"The Augmented Lagrange Multipler Method for Exact Recovery of
Corrupted Low-Rank Matrices" by Zhouchen Lin, Minming Chen, and Yi
Ma http://arxiv.org/abs/1009.5055

This is a straight implementation of that method with no
modifications.
"""
import numpy as np
from dimredu.lib.shrink import shrink
from dimredu.lib.project import Pi


def MC(D, Omega, mu_0=None, rho=None, epsilon1=None, epsilon2=None,
       maxIteration=100, verbose=True):
    """This is exactly the code from: "The Augmented Lagrange Multipler
    Method for Exact Recovery of Corrupted Low-Rank Matrices" by
    Zhouchen Lin, Minming Chen, and Yi Ma
    http://arxiv.org/abs/1009.5055

    Args:

       D: A partially observed matrix.  The entries where Omega is
           False do not matter to the final solution (but may effect
           the path the solver takes to get there.

       Omega: A matrix that shows where D is observed.  D is used
               where Omega is True and not used where Omega is False.

       mu_0: the intial value for the augmented Lagrangian
              parameter.  (optional, defaults to value from
              Lin-Chen-Ma)

       rho: the growth factor for the augmented Lagrangian
             parameter.  (optional, defaults to value from
             Lin-Chen-Ma)

       epsilon1: the first error criterion that controls for the
                  error in the constraint.  (optional, defaults to
                  value from Lin-Chen-Ma)

       epsilon2: the second error criterion that controls for the
                  convergence of the method.  (optional, defaults to
                  value from Lin-Chen-Ma)

       maxIterations: the maximum number of iterations to
                       use. (optional, defaults to 100)

       verbose: print out the convergence history. (optional,
                 defaults to True)

    Returns:
       A: the recovered matrix.

       E: the differences between the input matrix and the recovered
           matrix, so A+E=D.  (Note, generally E is not important,
           but Lin-Chen-Ma return it so we do the same here.)
    """
    assert D.shape == Omega.shape, 'D and Omega must have the same shape'
    D = np.matrix(D)

    m = D.shape[0]
    n = D.shape[1]

    if mu_0 is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.
        mu_0 = 1. / np.linalg.norm(D, ord=2)
    if rho is None:
        # The default values for mu_0 is from bottom of page 12 in
        # Lin-Chen-Ma.  The flatten here is important since the ord=1
        # norm from np.linalg.norm for a matrix is max(sum(abs(x),
        # axis=0)), which is *not* what we want.
        rho_s = np.linalg.norm(Omega.flatten(), ord=1) / (m * n)
        rho = 1.2172 + 1.8588 * rho_s
    if epsilon1 is None:
        # The default values for epsilon1 is from bottom of page
        # 12 in Lin-Chen-Ma.
        epsilon1 = 1e-7
    if epsilon2 is None:
        # The default values for epsilon2 is from bottom of page
        # 12 in Lin-Chen-Ma.
        epsilon2 = 1e-6

    Y_0 = np.matrix(np.zeros(D.shape))
    E_0 = np.matrix(np.zeros(D.shape))

    iteration = 0
    while True:
        # Break if we use too many interations
        iteration += 1
        if iteration > maxIteration:
            break

        # This is the mathematical content of the algorithm
        ###################################################
        # The full_matrices being true is required for non-square
        # matrices
        [U, S, VT] = np.linalg.svd(D - E_0 + Y_0 / mu_0, full_matrices=False)
        # This is current guess for the recovered matrix
        A_1 = U * np.diag(shrink(1. / mu_0, S)) * VT
        # The current error in the constraint
        E_1 = Pi(1 - Omega, D - A_1 + Y_0 / mu_0)
        # Update the Lagrange mutiplier
        Y_1 = Y_0 + mu_0 * (D - A_1 - E_1)
        ###################################################

        # If the method is convering well then increase mu_0 to focus
        # more on the constraint.
        if (mu_0 * np.linalg.norm(E_1 - E_0, ord='fro')
                / np.linalg.norm(D, ord='fro')) < epsilon2:
            mu_0 = rho * mu_0

        # stopping criterion from page 12 of Lin, Chen, and Ma.
        criterion1 = np.linalg.norm(
            D - A_1 - E_1, ord='fro') / np.linalg.norm(D, ord='fro')
        criterion2 = (np.min([mu_0, np.sqrt(mu_0)])
                      * np.linalg.norm(E_1 - E_0, ord='fro')
                      / np.linalg.norm(D, ord='fro'))

        if verbose:
            if iteration == 1:
                print()
                print("iteration criterion1 epsilon1 criterion2 epsilon2")
            if iteration % 10 == 0:
                print(('%9d %10.2e %8.2e %10.2e %8.2e' %
                      (iteration, criterion1, epsilon1, criterion2, epsilon2)))

        # If both error criterions are satisfied stop the algorithm
        if criterion1 < epsilon1 and criterion2 < epsilon2:
            break

        Y_0 = Y_1
        E_0 = E_1

    return [A_1, E_1]


def test_MC_square():
    n = 5
    np.random.seed(1234)
    U = np.matrix(np.random.random(size=[n, 1]))
    D = U * U.T
    Omega = np.zeros([n, n])
    tmp = np.random.uniform(size=[n, n])
    Omega[tmp < 0.7] = 1
    A, E = MC(np.multiply(Omega, D), Omega, maxIteration=200)
    print(A)
    print(D)
    assert np.allclose(D, A, atol=1e-1)


def test_MC_notSquare():
    m = 5
    n = 7
    np.random.seed(1234)
    U = np.matrix(np.random.random(size=[m, 1]))
    V = np.matrix(np.random.random(size=[n, 1]))
    D = U * V.T
    Omega = np.zeros([m, n])
    tmp = np.random.uniform(size=[m, n])
    Omega[tmp < 0.7] = 1
    A, E = MC(np.multiply(Omega, D), Omega, maxIteration=200)
    assert np.allclose(D, A, atol=1e-1)
