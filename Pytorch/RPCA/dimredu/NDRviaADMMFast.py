#! /usr/bin/env python
"""A non-linear dimension reduction solver that uses

 Paffenroth, R., Du Toit, P., Nong, R., Scharf, L., Jayasumana, A. P.,
 & Bandara, V. (2013). Space-time signal processing for distributed
 pattern detection in sensor networks. Selected Topics in Signal
 Processing, IEEE Journal of, 7(1), 38-49,

 Paffenroth, R. C., Nong, R., & Du Toit, P. C. (2013, September). On
 covariance structure in noisy, big data. In SPIE Optical Engineering+
 Applications (pp. 88570E-88570E). International Society for Optics and
 Photonics.  Chicago,

 "The Augmented Lagrange Multipler Method for Exact Recovery of
 Corrupted Low-Rank Matrices" by Zhouchen Lin, Minming Chen, and Yi
 Ma http://arxiv.org/abs/1009.5055

 Candes, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal
 component analysis?. Journal of the ACM (JACM), 58(3), 11.

 as a starting points.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from dimredu.lib.sparseSVDUpdate import sparseSVDUpdate
from dimredu.lib.minAPGFast import minAPGFast
from dimredu.lib.minNDRSD import minNDRSD
from dimredu.lib.projSVDToDist import projSVDToDist
from dimredu.lib.shrink import shrink
from dimredu.lib.nuclearNorm import nuclearNorm
from dimredu.lib.EDM import KFast, KAdjointFast


def objective(L, truncateK=0):
    return nuclearNorm(L, truncateK)


def vecLagrangian(EG, SD, MD, LDOmega, vecEpsilon, Yt, Yb, mu, truncateK,
                  debug=False):
    term1 = np.sum(EG[truncateK:])
    term2 = np.dot(Yt, np.abs(shrink(vecEpsilon, SD)))
    term3 = (mu / 2.) * (np.linalg.norm(np.abs(shrink(vecEpsilon,
                                                      SD)))**2)
    term4 = np.dot(Yb, MD - LDOmega - SD)
    term5 = (mu / 2.) * (np.linalg.norm(MD - LDOmega - SD)**2)
    if debug:
        print('vecLagrangian term1', term1)
        print('vecLagrangian term2', term2)
        print('vecLagrangian term3', term3)
        print('vecLagrangian term4', term4)
        print('vecLagrangian term5', term5)
        print('vecLagrangian sum', term1 + term2 + term3 + term4 + term5)

        term1a = np.sum(EG[truncateK:])
        term2a = (mu / 2.) * (np.linalg.norm(1. / mu * Yt +
                                             np.abs(shrink(vecEpsilon,
                                                           SD)))**2)
        term3a = (mu / 2.) * (np.linalg.norm(1. / mu * Yb +
                                             MD - LDOmega - SD)**2)
        term4a = -1. / (2 * mu) * np.linalg.norm(Yt)**2
        term5a = -1. / (2 * mu) * np.linalg.norm(Yb)**2
        print('other way', term1a + term2a + term3a + term4a + term5a)
        print('dropped consant Y', term1a + term2a + term3a)
        print('just  L', term1a + term3a)
        print('term1a', term1a)
        print('term3a', term3a)
        # print 'term3a "b"', 1./mu*Yb + MD - SD
        # print 'term3a "A"', - LDOmega

    return term1 + term2 + term3 + term4 + term5


def NDR(m, n, u, v, vecMD, vecEpsilon, maxRank,
        truncateK=0,
        mu=None, rho=None, epsilon1=None, epsilon2=None,
        tau=10.0,
        maxIteration=1000, verbose=True, hasWeave=True,
        grammianGuess=None, debug=False):
    """This is an optimized code based on:

    Paffenroth, R., Du Toit, P., Nong, R., Scharf, L., Jayasumana,
    A. P., & Bandara, V. (2013). Space-time signal processing for
    distributed pattern detection in sensor networks. Selected Topics
    in Signal Processing, IEEE Journal of, 7(1), 38-49

    and

    Paffenroth, R. C., Nong, R., & Du Toit, P. C. (2013,
    September). On covariance structure in noisy, big data. In SPIE
    Optical Engineering+ Applications
    (pp. 88570E-88570E). International Society for Optics and
    Photonics.  Chicago.

    Args:

       m, n: the full size of the input distance matrix MD.

       u, v, vecMD: the samples of MD as indices and values of a sparse
                    matrix.  All are one dimensional arrays.

       vecEpsilon: the pointwise error bounds for the distances

       maxRank: the maximum rank of MD to consider for completion.
                 (note, Lin-Che-Ma have a way to predict this, which
                 we are not using here)

       truncateK: how many singular values to ignore on the high end.
                  As opposed to 'maxRank', which ignores small singular
                  value, truncateK ignores the largest singular values.

       mu: the intial value for the augmented Lagrangian
            parameter.  (optional, defaults to value from
            Lin-Chen-Ma)

       rho: the growth factor for the augmented Lagrangian
             parameter.  (optional, defaults to value from
             Lin-Chen-Ma)

       epsilon1: the first error criterion that controls for the
                  error in the constraint.  (The idea for this is from
                  Lin-Chen-Ma)

       epsilon2: the second error criterion that controls for the
                  convergence of the method.  (The idea for this is from
                  Lin-Chen-Ma)

       tau: the Lipschitz constant for the APG solver.

       maxIterations: the maximum number of iterations to
                       use. (optional, defaults to 100)

       verbose - print out the convergence history. (optional,
                 defaults to True)

       grammianGuess - an initial guess for the solver.  Since the problem
               is convex this shouldn't matter as to the final
               answer, but may help convergence.

       dedug - turn on debugging output.

    Returns:

       U,E,VT: the SVD of the recovered low rank Grammian matrix.
    """
    assert len(u.shape) == 1, 'u must be one dimensional'
    assert len(v.shape) == 1, 'v must be one dimensional'
    assert len(vecMD.shape) == 1, 'vecMD must be one dimensional'
    assert 0 <= np.max(u) < m, 'An entry in u is invalid'
    assert 0 <= np.max(v) < n, 'An entry in v is invalid'

    # The minimum value of the observed entries of M
    minMD = np.min(vecMD)

    if epsilon1 is None:
        # The default values for epsilon1 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # FIXME: Are these good for NDR?
        epsilon1 = 1e-5
    if epsilon2 is None:
        # The default values for epsilon2 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # FIXME: Are these good for NDR?
        epsilon2 = 1e-4

    # We want to keep around a sparse matrix version of MDvec, but we need
    # to be careful about 0 values in MDvec, we don't want them to get
    # discarded when we convert to a sparse matrix!  In particular, we
    # are using the sparse matrix in a slightly odd way.  We intend
    # that MDvec stores both 0 and non-zero values, and that the entries
    # of MDvec which are not stored are *unknown* (and not necessarily 0).
    # Therefore, we process the input MDvec to make all 0 entries "small"
    # numbers relative to its smallest value.
    # Note, for distance matrices this is less of an issue, but we
    # do it regardless to keep thing consistent and safe.
    for i in range(len(vecMD)):
        if vecMD[i] == 0:
            vecMD[i] = minMD * epsilon1
        if vecEpsilon[i] == 0:
            vecEpsilon[i] = minMD * epsilon1

    # Create the required sparse matrices.  Note, u,v,vecMD might have
    # repeats, and that is ok since the sp.coo_matrix handles
    # that case, and we don't actually use d after here.
    MD = vecMD
    Epsilon = vecEpsilon

    # The SVD of the low rank part of the answer.
    if grammianGuess is None:
        UG = np.matrix(np.zeros([m, maxRank]))
        EG = np.zeros([maxRank])
        VGT = np.matrix(np.zeros([maxRank, n]))
    else:
        UG = grammianGuess['U']
        EG = grammianGuess['E']
        VGT = grammianGuess['VT']

    # Compute the largest singular values of D (assuming the
    # unobserved entries are 0.  I am not convinced this is
    # principled, but I believe it it what they do in the paper.
    MDsparse = csr_matrix(coo_matrix((MD, (u, v)), shape=(m, n)))
    dummy, ED0, dummy = sparseSVDUpdate(MDsparse, UG[:, 0],
                                        np.array([EG[0]]), VGT[0, :])

    if mu is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.  I believe that the use the
        # spectral norm of D (the largest singular value), where
        # the unobserved entries are assumed to be 0.
        # FIXME:  I am not sure this is principled.  I mean, why is 0 special?
        # I am pretty sure that I can break this with a inproperly scaled D.
        # FIXME: Are these good for eRPCA?
        mu = 1. / ED0[0]
    if rho is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # The flatten here is important since the ord=1 norm
        # from np.linalg.norm for a matrix is max(sum(abs(x), axis=0)), which
        # is *not* what we want.
        # FIXME: Are these good for eRPCA?
        rho_s = len(vecMD) / float(m * n)
        rho = 1.2172 + 1.8588 * rho_s

    # The sparse Lagrange multiplers
    Yt = MD * 0.0
    Yb = MD * 0.0

    # The sparse matrix S
    SD = MD * 0.0

    # The mapping of the Grammian LG to a distance matrix LD, and
    # then projected onto Omega.  This is not required
    # but is convenient to have.  We set it to be what we get
    # from the initial guess.
    fProjSVD = projSVDToDist
    LDOmega = fProjSVD(UG, EG, VGT, u, v, returnVec=True)

    # We keep the previous answer to check convergence
    # LDOmega0 = LDOmega.copy()

    # We also keep around a copy of the SVD of L, in case we want to
    # reset it.
    UG0 = UG.copy()
    EG0 = EG.copy()
    VGT0 = VGT.copy()

    # We also want this to check convergence
    partialFrobeniusMD = np.linalg.norm(MD)

    iteration = 0

    while True:
        # Break if we use too many interations
        iteration += 1
        if iteration > maxIteration:
            break

        # This is the mathematical content of the algorithm
        ###################################################

        ####################################
        # DEBUG ############################
        vecLagrangianValue0 = vecLagrangian(EG, SD, MD, LDOmega,
                                            Epsilon, Yt, Yb, mu,
                                            truncateK,
                                            debug=debug)
        # DEBUG ############################
        ####################################

        ####################################
        # TODO #############################
        ####################################
        # This should be something like the APG
        # algorithm on page 12 of papers/AccelProxForNucNorm.pdf.
        # You need equation (7) on page 3 of papers/AccelProxForNucNorm.pdf.
        # and the definition of the adjoint of the Grammian
        # to distance linear operator on page 5 of
        # papers/EDMhandbook.pdf
        ####################################
        # TODO #############################
        ####################################

        # Minimize the Lagrangian with respect to L with S fixed
        # NOTE: I think that getting the sign wrong here causes the
        # Lagrange multipler to make the solution worse!
        # X = -(-Yb/mu - MD + SD)
        X = Yb / mu + MD - SD

        [UG, EG, VGT] = minAPGFast(m, n,
                                   KFast, KAdjointFast,
                                   X, mu,
                                   tau=tau,
                                   truncateK=truncateK,
                                   debug=debug,
                                   guess={'U': UG0,
                                          'E': EG0,
                                          'VT': VGT0,
                                          'u': u,
                                          'v': v},
                                   # FIXME:  do something rational here.
                                   maxIter=3
                                   )

        # Compute the projection of L onto Omega
        LDOmega = fProjSVD(UG, EG, VGT, u, v, returnVec=True)

        ####################################
        # DEBUG ############################
        vecLagrangianValue1 = vecLagrangian(EG, SD, MD, LDOmega,
                                            Epsilon, Yt, Yb, mu,
                                            truncateK,
                                            debug=debug)
        if vecLagrangianValue1 > vecLagrangianValue0 + 1e-7:
            if verbose:
                print('Lagrangian went up after L minimization!')
                print('before', vecLagrangianValue0)
                print('after', vecLagrangianValue1)
                print('before-after', vecLagrangianValue0 -
                      vecLagrangianValue1)
                print('Perhaps you need to make tau bigger?')
            assert False, 'Lagrangian went up after L minimization!'
        # DEBUG ############################
        ####################################

        # Minimize the Lagrangian with respect to S with L fixed
        SD = minNDRSD(MD - LDOmega, Yt, Yb,
                      Epsilon, mu, debug=debug, guess=SD)

        ####################################
        # DEBUG ############################
        vecLagrangianValue2 = vecLagrangian(EG, SD, MD, LDOmega,
                                            Epsilon, Yt, Yb, mu,
                                            truncateK)
        if vecLagrangianValue2 > vecLagrangianValue1 + 1e-7:
            if verbose:
                print('Lagrangian went up after S minimization!')
                print('before', vecLagrangianValue0)
                print('after', vecLagrangianValue1)
            assert False, 'Lagrangian went up after S minimization!'
        # DEBUG ############################
        ####################################

        # Update the Lagrange mutipliers
        Yt = Yt + mu * np.abs(shrink(Epsilon, SD))
        Yb = Yb + mu * (MD - LDOmega - SD)

        ###################################################
        # If the method is converging well then increase mu_0 to focus
        # more on the constraint.  if
        # mu_0*np.linalg.norm(POA_1-POA_0,ord=2)/partialFrobeniusD <
        # epsilon2: Again, I don't know how to compute the spectral
        # norm of a partially observed matrix, so I replace with the
        # Froebenius norm on the observed entries FIXME: Attempt to
        # justify later.
        # tmp = np.linalg.norm(LDOmega0-LDOmega)
        # if mu*tmp/partialFrobeniusMD < epsilon2:
        #     mu = rho*mu
        if np.sum(EG[truncateK:]) < epsilon2:
            mu = rho * mu

        # stopping criterion from page 12 of Lin, Chen, and Ma.
        # criterion1 = np.linalg.norm(D-A_1-E_1,
        #                             ord='fro')/np.linalg.norm(D, ord='fro')
        # criterion1 = np.linalg.norm(D-A_1-(POA_1 - A_1),
        #                             ord='fro')/np.linalg.norm(D, ord='fro')
        # criterion1 = np.linalg.norm(D-POA_1,
        #                             ord='fro')/np.linalg.norm(D, ord='fro')
        # FIXME: I may need to justify the change from the full
        #        Froebenius norm to the partial one.
        tmp1 = np.linalg.norm(MD - LDOmega - SD)
        tmp2 = np.linalg.norm(shrink(Epsilon, SD), 1)
        criterion1 = (tmp1 + tmp2) / partialFrobeniusMD
        # criterion2 = np.min([mu_0, np.sqrt(mu_0)])*\
        #              np.linalg.norm(E_1-E_0,
        #                             ord='fro')/np.linalg.norm(D, ord='fro')
        # This is the one place where I depart from Lin-Chen-Ma.  The
        # stopping criterion there have right about equation uses A
        # and POA.  As I want the algorithm to be fast I ignore the A
        # part, since that would be O(mn)
        # FIXME:  Need to justify
        # FIXME: I may need to justify the change from the full
        #         Froebenius norm to the partial one.
        # criterion2 = (np.min([mu, np.sqrt(mu)]) *
        #               (np.linalg.norm(LDOmega0-LDOmega))/partialFrobeniusMD)
        criterion2 = np.sum(EG[truncateK:])

        if verbose:
            if iteration == 1:
                if verbose:
                    print()
                    print('criterion1 is the constraint')
                    print('criterion2 is the solution')
                    print('iteration criterion1 epsilon1 ', end=' ')
                    print('criterion2 epsilon2 rho      mu       objective')
            if iteration % 10 == 0:
                if verbose:
                    print('%9d %10.2e %8.2e ' % (iteration,
                                                 criterion1,
                                                 epsilon1), end=' ')
                    print('%10.2e %8.2e %8.2e %8.2e ' % (criterion2,
                                                         epsilon2,
                                                         rho,
                                                         mu), end=' ')
                    print('%9.2e' % np.sum(EG[truncateK:]))

        # If both error criterions are satisfied stop the algorithm
        if criterion1 < epsilon1 and criterion2 < epsilon2:
            if verbose:
                print('%9d %10.2e %8.2e ' % (iteration,
                                             criterion1,
                                             epsilon1), end=' ')
                print('%10.2e %8.2e %8.2e %8.2e ' % (criterion2,
                                                     epsilon2,
                                                     rho,
                                                     mu), end=' ')
                print('%9.2e' % np.sum(EG[truncateK:]))
            break

        # Keep around the old answer for convergence testing.
        # LDOmega0 = LDOmega.copy()
        UG0 = UG.copy()
        EG0 = EG.copy()
        VGT0 = VGT.copy()

    return [UG, EG, VGT, iteration]


def checkConstraints(U, L, A):
    worstE = 0.0
    worstI = 0
    worstJ = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if L[i, j] - A[i, j] > worstE:
                worstE = L[i, j] - A[i, j]
                worstI = i
                worstJ = j
            if A[i, j] - U[i, j] > worstE:
                worstE = A[i, j] - U[i, j]
                worstI = i
                worstJ = j
    if worstE > 0:
        print('worst constraint mismatch was %f at %d,%d' % (worstE,
                                                             worstI,
                                                             worstJ))


class problem1D(object):
    def __init__(self, n, r=0.5, large=1e1, seed=1234, gap=False):
        import sympy as sy
        from dimredu.lib.EDM import manifold, G, K, makeEpsilon

        self.n = n
        self.r = r
        self.large = large
        self.seed = seed
        # We want to have n points in 1-dimensional space
        self.kTrue = 1
        np.random.seed(seed)
        t = sy.symbols('t')
        x = sy.cos(t) * t
        y = sy.sin(t) * t

        m = manifold(x, y)
        if gap:
            s0 = np.linspace(0, np.pi / 4. - 0.2, n / 2)
            s1 = np.linspace(np.pi / 4. + 0.2, np.pi / 2, n / 2)
            s = np.concatenate((s0, s1))
        else:
            s = np.linspace(0, np.pi / 2, n)

        P = np.matrix(np.zeros([n, 2]))
        PTrue = np.matrix(np.zeros([n, 1]))
        for i in range(n):
            PTrue[i] = m.arclength(s[0], s[i])
            P[i, :] = m(s[i])

        self.PTrue = PTrue
        self.P = P

        self.G = G(P)
        self.GTrue = G(PTrue)

        self.D = K(self.G)
        self.DTrue = K(self.GTrue)

        # Small radius of curvature r means the curve can take tighter turns
        # and is easier to flatten.
        self.LConstraint, self.UConstraint =\
            makeEpsilon(self.D, r=r, large=large)

        self.M = (self.UConstraint + self.LConstraint) / 2
        self.Epsilon = (self.UConstraint - self.LConstraint) / 2

        u = []
        v = []
        vecMD = []
        vecEpsilon = []

        for i in range(n):
            for j in range(n):
                if i != j:
                    u.append(i)
                    v.append(j)
                    vecMD.append(self.M[i, j])
                    vecEpsilon.append(self.Epsilon[i, j])

        self.u = np.array(u)
        self.v = np.array(v)
        self.vecMD = np.array(vecMD)
        self.vecEpsilon = np.array(vecEpsilon)

    def setComputedG(self, G):
        from dimredu.lib.EDM import Gd, K
        self.GComputed = G
        self.PComputed = Gd(G)
        self.DComputed = K(G)

    def myPrint(self):
        np.set_printoptions(precision=5, suppress=False)
        print('computed grammian')
        print(self.GComputed)
        [U, E, VT] = np.linalg.svd(self.GComputed)
        print('  singular values')
        print('  ', E[:self.kTrue + 3])
        print('  nuclear norm')
        print('  ', np.sum(E))
        print('computed distances')
        print(self.DComputed)
        print('checkConstraints')
        checkConstraints(self.UConstraint, self.LConstraint,
                         self.DComputed)

    def plot(self):
        import matplotlib.pylab as py
        py.plot(np.array(self.P)[:, 0], np.array(self.P)[:, 1], 'ro')
        py.plot(np.array(self.PComputed)[:, 0],
                np.array(self.PComputed)[:, 1], 'go')


def test_compare():
    from dimredu.NDRviaCVXPy import NDR as CVXPyNDR
    import pytest
    pytest.xfail('NDRviaADMM fast is known to have Lagrangian going up')

    n = 5
    np.random.seed(1234)
    # We need the rank to be k+2 since it is an EDM
    problemCVXPY = problem1D(n)
    G1 = CVXPyNDR(problemCVXPY.LConstraint,
                  problemCVXPY.UConstraint,
                  problemCVXPY.kTrue + 2,
                  maxIter=100, type='Grammian',
                  returnType='Grammian')

    # Note, this returns the SVD of a Grammian
    problemADMM = problem1D(n)
    DU, DE, DVT, iter = NDR(problemADMM.n, problemADMM.n,
                            problemADMM.u, problemADMM.v,
                            problemADMM.vecMD,
                            problemADMM.vecEpsilon,
                            maxRank=problemADMM.n,
                            truncateK=problemADMM.kTrue,
                            maxIteration=1000,
                            debug=False)
    G2 = DU * np.diag(DE) * DVT
    assert np.allclose(G1, G2, atol=1e-1)


def check_medium():
    import matplotlib.pylab as py
    n = 50

    # Note, this returns the SVD of a Grammian
    problem = problem1D(n, gap=True)
    DU, DE, DVT, iter = NDR(problem.n, problem.n,
                            problem.u, problem.v,
                            problem.vecMD, problem.vecEpsilon,
                            maxRank=problem.n,
                            truncateK=problem.kTrue,
                            tau=100.0,
                            rho=1.05,
                            mu=0.0001,
                            # constraint
                            epsilon1=1e-5,
                            # solution
                            epsilon2=1e-5,
                            maxIteration=200,
                            debug=False)
    G = DU * np.diag(DE) * DVT
    problem.setComputedG(G)

    problem.myPrint()

    py.ion()
    py.figure(1)
    py.clf()
    problem.plot()
    py.show()


def check_many():
    n = 50
    import sys
    import pandas as pa

    # Note, this returns the SVD of a Grammian
    print('%15s, %15s, %15s, %15s' % ('tau', 'rho', 'mu', 'iter'))

    data = []
    for tau in 10**np.linspace(1.5, 2.5, 20):
        for rho in 1 + np.linspace(0, 0.5, 20):
            for mu in 10**np.linspace(-4, 1, 20):
                problem = problem1D(n, gap=True)
                try:
                    DU, DE, DVT, iter = NDR(problem.n, problem.n,
                                            problem.u, problem.v,
                                            problem.vecMD, problem.vecEpsilon,
                                            maxRank=problem.n,
                                            truncateK=problem.kTrue,
                                            tau=tau,
                                            rho=rho,
                                            mu=mu,
                                            # constraint
                                            epsilon1=1e-5,
                                            # solution
                                            epsilon2=1e-5,
                                            maxIteration=1000,
                                            debug=False,
                                            verbose=False)
                    G = DU * np.diag(DE) * DVT
                    problem.setComputedG(G)
                    data.append([tau, rho, mu, iter])
                    print('%15.2e, %15.2e, %15.2e, %15d' % (tau, rho,
                                                            mu, iter))
                except:
                    data.append([tau, rho, mu, -1])
                    print('%15.2e, %15.2e, %15.2e, %15d' % (tau, rho, mu, -1))
                sys.stdout.flush()
    dataFrame = pa.DataFrame(data, columns=['tau', 'rho', 'mu', 'iter'])
    dataFrame.to_hdf('log.h5', 'data')


def check_large():
    import matplotlib.pylab as py
    from dimredu.lib.EDM import Gd, G, K, makeEpsilon
    from dimredu.lib.nonlinearData import WPISwissRoll

    # size = 10000
    size = 100
    large = 1e2
    kTrue = 2

    np.random.seed(1234)
    u = np.random.random(size=[size])
    v = np.random.random(size=[size])

    uN, vN, x, y, z = WPISwissRoll(u, v)

    n = len(x)

    P = np.matrix([x, y, z]).T
    Y = G(P)
    D = K(Y)

    LConstraint, UConstraint = makeEpsilon(D, r=1., large=large)
    M = (UConstraint + LConstraint) / 2
    Epsilon = (UConstraint - LConstraint) / 2

    u = []
    v = []
    vecMD = []
    vecEpsilon = []

    for i in range(n):
        for j in range(n):
            if i != j and UConstraint[i, j] < large - 1:
                u.append(i)
                v.append(j)
                vecMD.append(M[i, j])
                vecEpsilon.append(Epsilon[i, j])

    print('D.shape', D.shape)
    print('D.shape[0]**2', D.shape[0]**2)
    print('len(vecMD)', len(vecMD))

    u = np.array(u)
    v = np.array(v)
    vecMD = np.array(vecMD)
    vecEpsilon = np.array(vecEpsilon)

    # Note, this returns the SVD of a Grammian
    DU, DE, DVT = NDR(n, n, u, v, vecMD, vecEpsilon,
                      maxRank=n,
                      truncateK=kTrue,
                      rho=1.01,
                      mu=2.0,
                      tau=1000.0,
                      maxIteration=100,
                      debug=False)

    GADMM = DU * np.diag(DE) * DVT

    print('Y')
    print(Y)
    [U, E, VT] = np.linalg.svd(Y)
    print('E')
    print(E)
    print('np.sum(E)')
    print(np.sum(E))

    print('GADMM')
    print(GADMM)
    [U, E, VT] = np.linalg.svd(GADMM)
    print('E')
    print(E)
    print('np.sum(E)')
    print(np.sum(E))

    print('K(GADMM)')
    print(K(GADMM))
    print('checkConstraints(U, L, K(GADMM))')
    checkConstraints(UConstraint, LConstraint, K(GADMM), tol=1e-1)

    PADMM = Gd(GADMM)

    # py.ion()
    py.figure(1)
    py.clf()
    py.plot(np.array(PADMM)[:, 0], np.array(PADMM)[:, 1], 'bo')
    py.show()


# def profile():
#     import cProfile
#     cProfile.run('test_large()', 'stats')


if __name__ == '__main__':
    test_compare()
    # check_medium()
    # check_many()
    # check_large()
    # profile()
