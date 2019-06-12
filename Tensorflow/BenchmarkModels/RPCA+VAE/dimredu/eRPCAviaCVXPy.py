import cvxpy
import numpy as np


def eRPCA(M, E, lam=None, eps=1e-6):
    """This is trivial implementation of RPCA using CVXPY

    Args

       M: A matrix which is presumed to be low rank except for sparse
           pertubations.

       E: A matrix of point-wise error bounds for M.

       lam: The :math:`\lambda` value that couples the low-rank and
             sparse parts.

       eps: The tolerance for the solver.

    Returns:

       L: The low rank matrix.

       S: The sparse pertubations.

    """
    assert M.shape == E.shape, 'M and E must have the same shape'

    m = M.shape[0]
    n = M.shape[1]

    if lam is None:
        lam = 1. / np.sqrt(np.max([m, n]))

    # Construct the problem.
    L = cvxpy.Variable(m, n)
    S = cvxpy.Variable(m, n)
    objective = cvxpy.Minimize(cvxpy.norm(L, 'nuc') +
                               lam * (cvxpy.norm(S, 1)))
    constraints = [cvxpy.abs(M - L - S) <= E]
    prob = cvxpy.Problem(objective, constraints)

    # prob.solve(solver='SCS', eps=1e-6)
    prob.solve(eps=1e-6)
    return [L.value, S.value]


def test_eRPCA_square():
    n = 5
    U = np.matrix(np.random.random(size=[n, 1]))
    LTrue = U * U.T

    tmp = np.random.uniform(size=[n, n])
    STrue = np.zeros([n, n])
    STrue[tmp > 0.9] = 1.

    E = np.ones([n, n]) * 1e-3

    L, S = eRPCA(LTrue + STrue, E)

    print('LTrue')
    print(LTrue)
    print('L')
    print(L)

    print('STrue')
    print(STrue)
    print('S')
    print(S)

    print('error in L')
    print((np.linalg.norm(L - LTrue)))
    print('error in S')
    print((np.linalg.norm(S - STrue)))


def test_eRPCA_notSquare():
    m = 5
    n = 7
    U = np.matrix(np.random.random(size=[m, 1]))
    V = np.matrix(np.random.random(size=[n, 1]))
    LTrue = U * V.T

    tmp = np.random.uniform(size=[m, n])
    STrue = np.zeros([m, n])
    STrue[tmp > 0.9] = 1.

    E = np.ones([m, n]) * 1e-3

    L, S = eRPCA(LTrue + STrue, E)

    print('LTrue')
    print(LTrue)
    print('L')
    print(L)

    print('STrue')
    print(STrue)
    print('S')
    print(S)

    print('error in L')
    print((np.linalg.norm(L - LTrue)))
    print('error in S')
    print((np.linalg.norm(S - STrue)))


if __name__ == '__main__':
    test_eRPCA_square()
    test_eRPCA_notSquare()
