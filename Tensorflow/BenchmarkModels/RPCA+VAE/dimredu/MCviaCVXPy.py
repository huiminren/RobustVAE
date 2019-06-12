#! /usr/bin/env python

import cvxpy
import numpy as np


def MC(D, Omega, epsilonSmall=1e-7, epsilonLarge=100.0):
    """This is trivial implementation of matrix completion using CVXPY

    Args:
       D: A partially observed matrix.  The entries where Omega is False
           do not matter to the final solution (but may effect the path the
           solver takes to get there.

       Omega: A matrix that shows where D is observed.  D is used
               where Omega is True and not used where Omega is False.

       epsilonSmall: The error value to use for "observed".

       epsilonLarge: The value to use for "unobserved".

    Returns:
       A - the recovered matrix.

    """
    assert D.shape == Omega.shape, 'D and Omega must have the same shape'

    epsilon = np.matrix(np.ones(D.shape)) * epsilonLarge
    epsilon[Omega == 1] = epsilonSmall

    # Construct the problem.
    L = cvxpy.Variable(D.shape[0], D.shape[1])

    objective = cvxpy.Minimize(cvxpy.norm(L, 'nuc'))

    constraints = [cvxpy.abs(D - L) <= epsilon]
    prob = cvxpy.Problem(objective, constraints)

    prob.solve(solver='SCS', eps=1e-6)
    return L.value


def test_MC_square():
    n = 5
    np.random.seed(1234)
    U = np.matrix(np.random.random(size=[n, 1]))
    D = U * U.T
    Omega = np.zeros([n, n])
    tmp = np.random.uniform(size=[n, n])
    Omega[tmp < 0.7] = 1
    A = MC(np.multiply(Omega, D), Omega)
    # Make sure that Omega has at least one 0
    assert not(np.all(Omega))
    # Test that the recovery is close to the original matrix.
    assert np.allclose(A, D, atol=1e-1)


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
    A = MC(np.multiply(Omega, D), Omega)
    # Make sure that Omega has at least one 0
    assert not(np.all(Omega))
    # Test that the recovery is close to the original matrix.
    assert np.allclose(A, D, atol=1e-1)
