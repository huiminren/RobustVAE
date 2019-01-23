#! /usr/bin/env python

import cvxpy
import numpy as np
import matplotlib.pylab as py
import sympy as sy
from dimredu.lib.EDM import G, Gd, K, Kd, makeEpsilon, manifold


def NDR(L, U, k, verbose=True, maxIter=1000, type='Distance',
        returnType='Distance'):
    n = L.shape[0]
    X0 = L.copy()
    # A dummy variable to enforce the semi-positive definiteness of
    # Moore-Penrose inverse.
    SPD = cvxpy.Semidef(n)

    if type != 'Distance':
        k = k - 2

    iteration = 0
    while True:
        [U0, E0, VT0] = np.linalg.svd(X0)
        A0 = U0[:, :k].T
        B0 = (VT0.T)[:, :k].T

        X = cvxpy.Variable(n, n)
        if type == 'Distance':
            # In this case X is a distance matrix
            objective = cvxpy.Minimize(cvxpy.norm(X, 'nuc') -
                                       cvxpy.trace(A0 * X * B0.T))
            constraints = [cvxpy.abs(X - (U + L) / 2.) <= (U - L) / 2.,
                           Kd(X, n) == SPD]
        else:
            # In this case X is a Grammian.
            objective = cvxpy.Minimize(cvxpy.norm(X, 'nuc') -
                                       cvxpy.trace(A0 * X * B0.T))
            e = np.matrix(np.ones([n, 1]))
            dX = cvxpy.diag(X)
            D = dX * e.T + e * dX.T - 2 * X
            constraints = [cvxpy.abs(D - (U + L) / 2.) <= (U - L) / 2.]

        prob = cvxpy.Problem(objective, constraints)
        prob.solve(solver='SCS', eps=1e-7)
        X1 = X.value

        if verbose and (iteration % 100 == 0):
            print('iteration: ', iteration, end=' ')
            print('error: ', np.linalg.norm(X1 - X0, 'fro'))

        if np.linalg.norm(X1 - X0, 'fro') < 1e-4:
            break
        else:
            X0 = X1

        if iteration >= maxIter:
            break

        iteration = iteration + 1
    if returnType == 'Distance':
        if type == 'Distance':
            return X1
        else:
            return K(X1)

    if returnType == 'Grammian':
        if type == 'Distance':
            return Kd(X1)
        else:
            return X1


# This is the explicit version of the objective
def objective(A, k):
    [U, E, VT] = np.linalg.svd(A)
    sum = 0
    for i in range(k, E.shape[0]):
        sum += E[i]
    return sum


# Return the singular values
def E(A):
    [U, E, VT] = np.linalg.svd(A)
    return E


def check():
    np.set_printoptions(precision=5, suppress=False)

    # We want to have n points in k dimensional space
    n = 5
    kTrue = 1
    np.random.seed(1)

    t = sy.symbols('t')
    x = sy.cos(t) * t
    y = sy.sin(t) * t

    m = manifold(x, y)
    s = np.linspace(0, np.pi / 2, n)
    P = np.matrix(np.zeros([n, 2]))
    PTrue = np.matrix(np.zeros([n, 1]))
    for i in range(n):
        PTrue[i] = m.arclength(s[0], s[i])
        P[i, :] = m(s[i])

    Y = G(P)
    YTrue = G(PTrue)

    D = K(Y)
    DTrue = K(YTrue)

    # Small radius of curvature r means the curve can take tighter turns
    # and is easier to flatten.
    L, U = makeEpsilon(D, r=0.5, large=1e1)
    # L, U = makeEpsilon(D, r=1.0, large=1e1)
    # L = D
    # U = DTrue

    # We need the rank to be k+2 since it is an EDM
    DHat = NDR(L, U, kTrue + 2, maxIter=100, type='Grammian')

    print('L')
    print(L)
    print('D')
    print(D)
    print('DTrue')
    print(DTrue)
    print('U')
    print(U)

    # The +2 is since we are looking at D while rTrue
    # is the target dimension of Y
    print('objective before')
    print(objective(D, kTrue + 2))
    print(E(D)[:5])
    print('objective after')
    print(objective(DHat, kTrue + 2))
    print(E(DHat)[:5])
    print('objective true')
    print(objective(DTrue, kTrue + 2))
    print(E(DTrue)[:5])

    py.ion()
    py.figure(1).clf()

    f, ax = py.subplots(2, 3, num=1)

    myAx = ax[0, 0]
    myAx.cla()
    # myAx.set_title('test')
    myAx.axis('equal')
    myAx.plot(np.array(P)[:, 0], np.array(P)[:, 1], 'bo-')
    myAx.plot(np.array(Gd(Y))[:, 0], np.array(Gd(Y))[:, 1], 'r.-')
    myAx.plot(np.array(Gd(YTrue))[:, 0], np.array(Gd(YTrue))[:, 1], 'g.-')
    myAx.legend(['P', 'Gd(Y)', 'Gd(YTrue)'])

    myAx = ax[0, 1]
    myAx.cla()
    myAx.plot(np.array(U).flatten(),  'k')
    myAx.plot(np.array(L).flatten(), 'k')
    myAx.plot(np.array(D).flatten(), 'co')
    myAx.plot(np.array(DHat).flatten(), 'm')
    myAx.plot(np.array(DTrue).flatten(), 'yo')
    # myAx.set_ylim([0, 1])
    myAx.legend(['U', 'L', 'D', 'DHat', 'DTrue'])

    myAx = ax[1, 0]
    myAx.cla()
    myAx.plot(E(Y),  'r')
    myAx.plot(E(YTrue), 'g')
    myAx.legend(['E(Y)', 'E(YTrue)'])

    myAx = ax[1, 1]
    myAx.cla()
    myAx.plot(E(D), 'r')
    myAx.plot(E(DTrue), 'g')
    myAx.plot(E(DHat), 'bo')
    myAx.legend(['E(D)', 'E(DTrue)', 'E(DHat)'])

    myAx = ax[0, 2]
    myAx.cla()
    myAx.axis('equal')
    myAx.plot(np.array(Gd(Kd(D, n)))[:, 0],
              np.array(Gd(Kd(D, n)))[:, 1], 'r.-')
    myAx.plot(np.array(Gd(Kd(DTrue, n)))[:, 0],
              np.array(Gd(Kd(DTrue, n)))[:, 1], 'g.-')
    myAx.plot(np.array(Gd(Kd(DHat, n)))[:, 0],
              np.array(Gd(Kd(DHat, n)))[:, 1], 'b+-')
    myAx.legend(['Gd(Kd(D, n))', 'Gd(Kd(DTrue, n))', 'Gd(Kd(DHat, n))'])

    myAx = ax[1, 2]
    myAx.cla()
    myAx.plot(np.array(D).flatten(), 'r')
    myAx.plot(np.array(DTrue).flatten(), 'g')
    myAx.plot(np.array(DHat).flatten(), 'bo')
    myAx.legend(['D', 'DTrue', 'DHat'])

    py.show()

    print(Kd(DHat, n))
    print(np.linalg.eigvals(Kd(DTrue, n)))
    # Ok, the rank of an EDM is at most rank of Grammian plus 2,
    # and generically has exactly that rank.
    # unfortunately, here it looks I have I have a rank 3 EDM that
    # when I map back to a Grammian is rank 2!  That is wierd...
    # but I guess it does not contradict the theorem.
    print(np.linalg.eigvals(Kd(DHat, n)))
    print(np.linalg.eigvals(Kd(D, n)))


if __name__ == '__main__':
    check()
