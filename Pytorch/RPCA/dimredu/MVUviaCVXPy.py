#! /usr/bin/env python

import cvxpy
import numpy as np


def MVU(Y, Omega, p):
    """This is an implementation of Maximum Variance Unfolding using CVXPY;

    Args:
       Y - A m by n matrix of n data points in m dimensional space.

       Omega - A 0-1 matrix that encodes the distances to obsere.
          1 for a distance to enforce and 0 for a free distance

       p -  The dimension of the low dimensional space.

    Returns:
       XHat - .A p by n matrix of n data points in p dimensional space.

    """
    assert isinstance(Y, np.matrix), 'Y must be a matrix'
    n = Y.shape[1]

    assert Omega.shape[0] == Omega.shape[0] == n, 'Omega wrong size'

    Sy = Y.T * Y

    K = cvxpy.Semidef(n)

    # We want to maximize the trace(K), and that is the same thing as
    # minimizing -trace(K).  Note, trace(K) is affine, so both trace(K)
    # and -trace(K) are convex.
    objective = cvxpy.Minimize(-cvxpy.trace(K))

    constraints = []
    # constraints.append(cvxpy.sum_entries(K) == 0)
    for i in range(n):
        for j in range(n):
            if Omega[i, j] == 1:
                constraints.append(K[i, j] == Sy[i, j])

    prob = cvxpy.Problem(objective, constraints)

    result = prob.solve(solver='SCS', eps=1e-6)
    U, E, UT = np.linalg.svd(np.matrix(K.value))
    return np.diag(np.sqrt(E[:p])) * UT[:p, :]


# # This is the explicit version of the objective
# def objective(A, k):
#     [U, E, VT] = np.linalg.svd(A)
#     sum = 0
#     for i in range(k, E.shape[0]):
#         sum += E[i]
#     return sum


# Return the singular values
def E(A):
    [U, E, VT] = np.linalg.svd(A)
    return E


def check():
    from dimredu.lib.EDM import K
    np.set_printoptions(precision=5, suppress=False)

    m = 3
    n = 10
    p = 2

    Y = np.matrix(np.random.random(size=[m, n]))
    Omega = np.matrix(np.zeros([n, n]))
    D = K(Y.T * Y)

    for i in range(n):
        for j in range(n):
            if D[i, j] < 0.5:
                Omega[i, j] = 1

    X = MVU(Y, Omega, p)

    print(Y)
    print(X)
    print(Omega)


def check1D():
    import sympy as sy
    from dimredu.lib.EDM import G, Gd, K, Kd, makeEpsilon, manifold
    import matplotlib.pylab as py

    np.set_printoptions(precision=5, suppress=False)

    # We want to have n points in k dimensional space
    n = 5
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
    Omega = np.matrix(np.zeros([n, n]))
    for i in range(n):
        for j in range(n):
            if D[i, j] < 0.5:
                Omega[i, j] = 1
    print(Omega)
    XHat = MVU(P.T, Omega, 1)
    DHat = K(XHat.T * XHat)

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


def check2D():
    from dimredu.lib.EDM import K
    from dimredu.lib.nonlinearData import swissRoll
    import mayavi.mlab as ml
    import matplotlib.pylab as py

    np.set_printoptions(precision=5, suppress=False)

    # We want to have n points in k dimensional space
    n = 100
    d = 1.0

    u = np.random.random(size=[n]) * 0.7
    v = np.random.random(size=[n])

    uN, vN, x, y, z = swissRoll(u, v)

    Y = np.matrix(np.empty([3, n]))
    Y[0, :] = x
    Y[1, :] = y
    Y[2, :] = z
    G = Y.T * Y
    D = K(G)

    print(('max distance', np.max(D)))
    print(('mean distance', np.mean(D)))
    print(('min distance', np.min(D)))

    Omega = np.matrix(np.zeros([n, n]))
    for i in range(n):
        for j in range(n):
            if D[i, j] < d:
                Omega[i, j] = 1

    XHat = MVU(Y, Omega, 2)
    GHat = XHat.T * XHat

    for i in range(n):
        for j in range(i + 1, n):
            if Omega[i, j]:
                print((i, j, G[i, j], GHat[i, j]))

    ml.figure(figure=ml.gcf(), size=(2000, 2000))
    ml.clf()

    ml.points3d(x, y, z, uN,
                mode='sphere',
                scale_factor=0.1, scale_mode='none')

    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            if Omega[i, j]:
                connections.append([i, j])
    connections = np.vstack(connections)

    # Create the points
    src = ml.pipeline.scalar_scatter(x, y, z)

    # Connect them
    src.mlab_source.dataset.lines = connections

    # The stripper filter cleans up connected lines
    lines = ml.pipeline.stripper(src)

    # Finally, display the set of lines
    ml.pipeline.surface(lines, colormap='Accent',
                        line_width=1, opacity=.4)

    py.ion()
    py.scatter(XHat[0, :], XHat[1, :], c=uN)
    py.draw()

    ml.show()


if __name__ == '__main__':
    # check()
    check1D()
    # check2D()
