import numpy as np
import sympy as sy
from scipy.sparse import coo_matrix, csr_matrix, diags
from dimredu.lib.projSVDToDist import projSVDToDist

# We use the notation, definitions, and ideas from
# "Euclidean Distance Matrices and Applications", Nathan Krislock and
# Henry Wolkowicz


# Turn points into a Gram matrix
def G(P):
    Pm = P - np.mean(P, axis=0)
    return Pm * Pm.T


# Turn a Gram matrix into points
def Gd(Y, r=2):
    [U, E, V] = np.linalg.svd(Y)
    output = []
    for i in range(r):
        # Not the multplication by np.sqrt(E[i]) is important,
        # otherwise the flat ones are not flat!
        output.append(np.array(U)[:, i] * np.sqrt(E[i]))
    return np.matrix(output).T


# We have the mappings to EDM and back
# Directly from the positions
def KFromP(P):
    n = P.shape[0]
    D = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            x = np.array(P)[i, :]
            y = np.array(P)[j, :]
            D[i, j] = np.dot(x - y, x - y)
    return D


# From Grammian to distance
def K(Y):
    n = Y.shape[0]
    e = np.matrix(np.ones([n, 1]))
    dY = np.matrix(np.diag(Y)).T
    return dY * e.T + e * dY.T - 2 * Y


# From Grammian to a projected distance matrix using a fast algorithm
# for a Grammian matrix represented as an SVD
def KFast(U, E, VT, u, v):
    return projSVDToDist(U, E, VT, u, v, returnVec=True)


# The adjoint operator for K.  This looks a lot
# like the Graph Laplacian!
def KAdjoint(D, symmetric=True):
    # Note, there is a symmetrix version of the adjoint and
    # a non-symmetric version of the adjoint.  Distance
    # matrices are symmetric so that is the default.  We use
    # the non-symmetric version for some testing purposes.
    e = np.matrix(np.ones([D.shape[1], 1]))
    eT = np.matrix(np.ones([1, D.shape[1]]))
    if symmetric:
        return 2 * (np.diag(np.array(D * e)[:, 0]) - D)
    else:
        return np.diag(np.array(D * e)[:, 0]) + \
            np.diag(np.array(eT * D)[0, :]) - 2 * D


# The adjoint operator for K, using a fast algorithm
# for the adjoint of a projection composed with
# the adjoint of K.
def KAdjointFast(x, u, v, m, n):
    e = np.matrix(np.ones([n, 1]))
    eT = np.matrix(np.ones([1, m]))
    X = csr_matrix(coo_matrix((x, (u, v)), shape=(m, n)))
    rowSums = diags(np.transpose(np.array(X * e)), [0])
    colSums = diags(np.array(eT * X), [0])
    return rowSums + colSums - 2 * X


# From distance to grammian
def Kd(D, n):
    J = np.eye(n) - np.ones([n, n]) / n
    return -0.5 * J * D * J


# This is the half-circle epsilon
def makeEpsilon(D, r, large=1e2):
    L = np.zeros(D.shape)
    U = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i, j] > 2 * r:
                L[i, j] = D[i, j]
                U[i, j] = large
            else:
                L[i, j] = D[i, j]
                # Remember, D is the distane squared!
                # Check that everywhere!
                d = np.sqrt(D[i, j])
                # This is the distance you get as a function of R
                theta = np.arcsin(d / (2 * r))
                U[i, j] = (2 * theta * r)**2
    return [L, U]


# A symbolic manifold that lets us have all the details
# we really want.  Mainly for testing.
class manifold(object):
    def __init__(self, x, y):
        t = sy.symbols('t')
        self.x = x
        self.y = y
        xp = sy.diff(x, t)
        yp = sy.diff(y, t)
        xpp = sy.diff(x, t, t)
        ypp = sy.diff(y, t, t)

        # The arclength on the manifold
        self.al = sy.sqrt(xp**2 + yp**2)
        # The radius of curvature of the manifold at each point
        self.r = ((xp**2 + yp**2)**(3 / 2)) / (xp * ypp - yp * xpp)

    def __call__(self, s):
        t = sy.symbols('t')
        return np.array([self.x.subs(t, s), self.y.subs(t, s)])

    def arclength(self, a, b):
        t = sy.symbols('t')
        return sy.N(sy.Integral(self.al, (t, a, b)))

    def radiusOfCurvature(self, s):
        t = sy.symbols('t')
        return self.r.subs(t, s)


def test_Sympy():
    t = sy.symbols('t')

    x = sy.cos(t) * t
    y = sy.sin(t) * t

    m = manifold(x, y)

    assert (m.arclength(0.0, 0.5) + m.arclength(0.5, 1.0) ==
            m.arclength(0.0, 1.0))

    assert m.radiusOfCurvature(0.8) < m.radiusOfCurvature(0.9)


def test_Adjoint():
    def Kvv(g, n):
        G = np.reshape(g, [n, n])
        D = K(G)
        d = np.reshape(np.array(D), [n * n])
        return d

    def KAdjointvv(d, n):
        D = np.reshape(d, [n, n])
        G = KAdjoint(D, symmetric=False)
        g = np.reshape(np.array(G), [n * n])
        return g

    n = 3
    MK = np.zeros([n * n, n * n])
    MKT = np.zeros([n * n, n * n])
    for i in range(n * n):
        e = np.zeros([n * n])
        e[i] = 1.
        MK[:, i] = Kvv(e, n)
        MKT[:, i] = KAdjointvv(e, n)

    assert np.all(np.transpose(MK) == MKT)


if __name__ == '__main__':
    test_Sympy()
    # test_Adjoint()
