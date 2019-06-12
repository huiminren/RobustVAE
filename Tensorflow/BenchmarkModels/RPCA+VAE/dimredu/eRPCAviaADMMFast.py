"""A eRPCA solver that uses

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
 component analysis?, Journal of the ACM (JACM), 58(3), 11.

 as a starting points.
"""

import numpy as np
import scipy.sparse as sp
from dimredu.lib.sparseSVDUpdate import sparseSVDUpdate
from dimredu.lib.sparseFrobeniusNorm import sparseFrobeniusNorm
from dimredu.lib.minNucPlusFrob import minNucPlusFrob
from dimredu.lib.minShrink1Plus2Norm import minShrink1Plus2Norm
from dimredu.lib.projSVD import projSVD
from dimredu.lib.shrink import shrink


def objective(L, S, Epsilon, lam, truncateK=0):
    U, E, VT = np.linalg.svd(L)
    return (np.sum(E[truncateK:]) +
            lam * np.linalg.norm(shrink(np.array(Epsilon).flatten(),
                                        np.array(S).flatten()), 1))


def vecLagrangian(E, S, M, LOmega, vecEpsilon, Y, mu, lam, truncateK):
    term1 = np.sum(E[truncateK:])
    term2 = lam * np.linalg.norm(shrink(vecEpsilon, S.data), 1)
    term3 = np.dot(Y.data, M.data - LOmega.data - S.data)
    term4 = (mu / 2) * sparseFrobeniusNorm(M - LOmega - S)**2
    return term1 + term2 + term3 + term4


def eRPCA(m, n, u, v, vecM, vecEpsilon, maxRank,
          lam=None, mu=None, rho=None, epsilon1=None, epsilon2=None,
          truncateK=0,
          maxIteration=1000, verbose=True, hasWeave=True):
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

       m, n: the full size of the input matrix M.

       u, v, vecM: the samples of M as indices and values of a sparse
                    matrix.  All are one dimensional arrays.

       vecEpsilon: the pointwise error bounds.

       maxRank: the maximum rank of M to consider for completion.
                 (note, Lin-Che-Ma have a way to predict this, which
                 we are not using here)

       lam: the value of the coupling constant between L and S

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

       maxIterations: the maximum number of iterations to
                       use. (optional, defaults to 100)

       verbose - print out the convergence history. (optional,
                 defaults to True)

    Returns:

       U,E,VT: the SVD of the recovered low rank matrix.

       S: A sparse matrix.
    """
    assert len(u.shape) == 1, 'u must be one dimensional'
    assert len(v.shape) == 1, 'v must be one dimensional'
    assert len(vecM.shape) == 1, 'vecM must be one dimensional'
    assert 0 <= np.max(u) < m, 'An entry in u is invalid'
    assert 0 <= np.max(v) < n, 'An entry in v is invalid'

    # The minimum value of the observed entries of M
    minM = np.min(vecM)

    if epsilon1 is None:
        # The default values for epsilon1 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # FIXME: Are these good for eRPCA?
        epsilon1 = 1e-5
    if epsilon2 is None:
        # The default values for epsilon2 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # FIXME: Are these good for eRPCA?
        epsilon2 = 1e-4

    # We want to keep around a sparse matrix version of Mvec, but we need
    # to be careful about 0 values in Mvec, we don't want them to get
    # discarded when we convert to a sparse matrix!  In particular, we
    # are using the sparse matrix in a slightly odd way.  We intend
    # that Mvec stores both 0 and non-zero values, and that the entries
    # of Mvec which are not stored are *unknown* (and not necessarily 0).
    # Therefore, we process the input Mvec to make all 0 entries "small"
    # numbers relative to its smallest value.
    for i in range(len(vecM)):
        if vecM[i] == 0:
            vecM[i] = minM * epsilon1
        if vecEpsilon[i] == 0:
            vecEpsilon[i] = minM * epsilon1

    # Create the required sparse matrices.  Note, u,v,d might have
    # repeats, and that is ok since the sp.coo_matrix handles
    # that case, and we don't actually use d after here.
    M = sp.csc_matrix(sp.coo_matrix((vecM, (u, v)),
                                    shape=[m, n]))
    Epsilon = sp.csc_matrix(sp.coo_matrix((vecEpsilon, (u, v)),
                                          shape=[m, n]))

    # The SVD of the low rank part of the answer.
    U = np.matrix(np.zeros([m, maxRank]))
    E = np.zeros([maxRank])
    VT = np.matrix(np.zeros([maxRank, n]))

    # Compute the largest singular values of D (assuming the
    # unobserved entries are 0.  I am not convinced this is
    # principled, but I believe it it what they do in the paper.
    dummy, E0, dummy = sparseSVDUpdate(M, U[:, 0], np.array([E[0]]), VT[0, :])

    if mu is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.  I believe that the use the
        # spectral norm of D (the largest singular value), where
        # the unobserved entries are assumed to be 0.
        # FIXME:  I am not sure this is principled.  I mean, why is 0 special?
        # I am pretty sure that I can break this with a inproperly scaled D.
        # FIXME: Are these good for eRPCA?
        mu = 1. / E0[0]
    if rho is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # The flatten here is important since the ord=1 norm
        # from np.linalg.norm for a matrix is max(sum(abs(x), axis=0)), which
        # is *not* what we want.
        # FIXME: Are these good for eRPCA?
        rho_s = len(vecM) / float(m * n)
        rho = 1.2172 + 1.8588 * rho_s
    if lam is None:
        # FIXME:  Double check this with Candes "RPCA?"
        lam = 1. / np.sqrt(np.max([m, n]))

    # The sparse Lagrange multiplers
    Y = M * 0.0

    # The sparse matrix S
    S = M * 0.0

    # The projection of L onto Omega.  This is not required
    # but is convenient to have.
    LOmega = M * 0.0

    # We keep the previous answer to check convergence
    LS0 = S + LOmega

    # We also want this to check convergence
    partialFrobeniusM = sparseFrobeniusNorm(M)

    iteration = 0

    while True:
        # Break if we use too many interations
        iteration += 1
        if iteration > maxIteration:
            break

        # This is the mathematical content of the algorithm
        ###################################################

        # # DEBUG ############################
        # print 'lagrangian before min L with S fixed',
        # vecLagrangianValue0 = vecLagrangian(E, S, M, LOmega,
        #                                     Epsilon.data, Y, mu, lam,
        #                                     truncateK=truncateK)
        # # DEBUG ############################

        # Minimize the Lagrangian with respect to L with S fixed
        [U, E, VT] = minNucPlusFrob(Y / mu + M - LOmega - S, U, E, VT,
                                    mu, truncateK=truncateK)

        # If the smallest signular value we compute is too large
        # then we might have maxRank too small (and run into error problems).
        # We check that here.
        if (E[0] > epsilon2) and (E[-1] / E[0] > epsilon2):
            print('Smallest singular value may be too big, consider')
            print('increasing maxRank.  This will make the solver slower,')
            print('but improve convergence')

        # Compute the project of L onto Omega
        LOmega = projSVD(U, E, VT, u, v)

        # FIXME  I need a good test here.  Because of my approximations
        #        I suspect that the Lagrangian can go up based upon
        #        the above minimization.
        # # DEBUG ############################
        # print 'lagrangian before min S with L fixed',
        # vecLagrangianValue1 = vecLagrangian(E, S, M, LOmega,
        #                                     Epsilon.data, Y, mu, lam,
        #                                     truncateK=truncateK)
        # assert vecLagrangianValue1 <= vecLagrangianValue0, \
        #    'Lagrangian went up!'
        # # DEBUG ############################

        # Minimize the Lagrangian with respect to S with L fixed
        S.data = minShrink1Plus2Norm(Y.data / mu + M.data - LOmega.data,
                                     Epsilon.data, lam, mu)

        # # DEBUG ############################
        # print 'lagrangian before Y update',
        # vecLagrangianValue2 = vecLagrangian(E, S, M, LOmega,
        #                                     Epsilon.data, Y, mu, lam,
        #                                     truncateK=truncateK)
        # assert vecLagrangianValue2 <= vecLagrangianValue1, \
        #    'Lagrangian went up!'
        # # DEBUG ############################

        # Update the Lagrange mutiplier
        Y.data = Y.data + mu * (M.data - S.data - LOmega.data)

        ###################################################
        # If the method is converging well then increase mu_0 to focus
        # more on the constraint.  if
        # mu_0*np.linalg.norm(POA_1-POA_0,ord=2)/partialFrobeniusD <
        # epsilon2: Again, I don't know how to compute the spectral
        # norm of a partially observed matrix, so I replace with the
        # Froebenius norm on the observed entries FIXME: Attempt to
        # justify later.
        if mu * sparseFrobeniusNorm(LS0 - LOmega - S) / partialFrobeniusM < epsilon2:
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
        criterion1 = sparseFrobeniusNorm(M - LOmega - S) / partialFrobeniusM
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
        criterion2 = (np.min([mu, np.sqrt(mu)]) *
                      sparseFrobeniusNorm(LS0 - LOmega - S) / partialFrobeniusM)

        if verbose:
            if iteration == 1:
                print()
                print('criterion1 is the constraint')
                print('criterion2 is the solution')
                print('iteration criterion1 epsilon1 ', end='')
                print('criterion2 epsilon2 rho      mu')
            if iteration % 10 == 0:
                print('%9d %10.2e %8.2e ' % (iteration,
                                             criterion1,
                                             epsilon1), end='')
                print('%10.2e %8.2e %8.2e %8.2e' % (criterion2,
                                                    epsilon2,
                                                    rho,
                                                    mu))

        # If both error criterions are satisfied stop the algorithm
        if criterion1 < epsilon1 and criterion2 < epsilon2:
            if verbose:
                print('%9d %10.2e %8.2e ' % (iteration,
                                             criterion1,
                                             epsilon1), end='')
                print('%10.2e %8.2e %8.2e %8.2e' % (criterion2,
                                                    epsilon2,
                                                    rho,
                                                    mu))
            break

        # Keep around the old answer for convergence testing.
        LS0 = LOmega + S

    S.data = shrink(Epsilon.data, S.data)

    return [U, E, VT, S]


def test_compare():
    from dimredu.eRPCAviaCVXPy import eRPCA as eRPCACVXPy
    np.random.seed(1234)
    m = 3
    n = 4
    U = np.matrix(np.random.random(size=[m, 1]))
    V = np.matrix(np.random.random(size=[n, 1]))
    L = U * V.T

    Omega = np.zeros([m, n])
    tmp = np.random.uniform(size=[m, n])
    Omega[tmp < 0.7] = 1

    Epsilon = np.zeros([m, n])

    S = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if Omega[i, j] == 1:
                # In this case we observe Epsilon
                Epsilon[i, j] = 1e-3
                if np.random.random() < 0.2:
                    # For some observed Epsilons we have  S
                    S[i, j] = 1.
            else:
                # In this case we do not observe Epsilon
                Epsilon[i, j] = 100.0
    M = np.multiply(Omega, L) + S

    LCVXPy, SCVXPy = eRPCACVXPy(M, Epsilon)

    u = []
    v = []
    vecM = []
    vecEpsilon = []
    for i in range(m):
        for j in range(n):
            if Omega[i, j] == 1:
                u.append(i)
                v.append(j)
                vecM.append(M[i, j])
                vecEpsilon.append(Epsilon[i, j])
    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)

    [U, E, VT, S] = eRPCA(m, n, u, v, vecM, vecEpsilon, 4, rho=1.01, mu=0.1)
    LFast = U * np.diag(E) * VT
    SFast = S.todense()

    print('Omega')
    print(Omega)

    print('LCVXPy')
    print(LCVXPy)
    print('LFast')
    print(LFast)

    print('SCVXPy')
    print(SCVXPy)
    print('SFast')
    print(SFast)

    lam = 1. / np.sqrt(np.max([m, n]))
    print('CVXPy objective')
    print(objective(LCVXPy, SCVXPy, Epsilon, lam))
    print('Fast objective')
    print(objective(LFast, SFast, Epsilon, lam))


def test_large():
    np.random.seed(1234)
    # m = 4000
    # n = 8000
    m = 400
    n = 800
    obs = m * 10

    U = np.matrix(np.random.random(size=[m, 1]))
    V = np.matrix(np.random.random(size=[n, 1]))

    u = []
    v = []
    vecM = []
    vecEpsilon = []

    print('starting to generate data')

    # Note, this code can create duplicates, but that is ok
    # for  coo_matrix
    for k in range(obs):
        i = np.random.randint(m)
        j = np.random.randint(n)
        u.append(i)
        v.append(j)
        vecEpsilon.append(1e-3)
        Mij = float(U[i, :] * (V.T)[:, j])
        # Sometimes put in an S
        if np.random.uniform() < 0.1:
            Mij += 1
        vecM.append(Mij)

    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)

    print('total number of entries: ', m * n)
    print('observed entries:', len(u))

    print('starting solve')
    [U, E, VT, S] = eRPCA(m, n, u, v, vecM, vecEpsilon,
                          4, rho=None, mu=None)

    lam = 1. / np.sqrt(np.max([m, n]))
    print(np.sum(E) + lam * np.linalg.norm(shrink(vecEpsilon, S.data), 1))


def profile():
    import cProfile
    cProfile.run('test_large()', 'stats')


if __name__ == '__main__':
    # test_compare()
    test_large()
    # profile()
