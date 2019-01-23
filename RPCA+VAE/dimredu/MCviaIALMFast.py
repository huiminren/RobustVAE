#! /usr/bin/env python
"""A matrix completion solver the implements Algorithm 6 (Matrix
Completion via Inexact ALM Method) from

"The Augmented Lagrange Multipler Method for Exact Recovery of
 Corrupted Low-Rank Matrices"
by Zhouchen Lin, Minming Chen, Leqin Wu, and Yi Ma
http://arxiv.org/abs/1009.5055

This version is optimized for partially observed matrices.
"""

import numpy as np
import scipy.sparse as sp
from dimredu.lib.sparseSVDUpdate import sparseSVDUpdate
from dimredu.lib.projSVD import projSVD
from dimredu.lib.sparseFrobeniusNorm import sparseFrobeniusNorm
from dimredu.lib.minNucPlusFrob import minNucPlusFrob


def MC(m, n, u, v, d, maxRank, mu_0=None, rho=None, epsilon1=None,
       epsilon2=None, maxIteration=100, verbose=True, hasWeave=True):
    """ This is an optimized code from:
    "The Augmented Lagrange Multipler Method for Exact Recovery
     of Corrupted Low-Rank Matrices"
    by Zhouchen Lin, Minming Chen, and Yi Ma
    http://arxiv.org/abs/1009.5055

    Args:

        m, n: the full size of D.

        u, v, d: the samples of D as indices and values of a sparse matrix.
            All are one dimensional arrays.

        maxRank: the maximum rank of D to consider for completion.
          (note, Lin-Che-Ma have a way to predict this,
           which we are not using here)

        mu_0: the intial value for the augmented Lagrangian parameter.
          (optional, defaults to value from
               Lin-Chen-Ma)

        rho: the growth factor for the augmented Lagrangian parameter.
          (optional, defaults to value from Lin-Chen-Ma)

        epsilon1: the first error criterion that controls for the error in
          the constraint.  (optional, defaults to value from Lin-Chen-Ma)

        epsilon2: the second error criterion that controls for the convergence
          of the method. (optional, defaults to value from Lin-Chen-Ma)

        maxIterations: the maximum number of iterations to use.
          (optional, defaults to 100)

        verbose: print out the convergence history.
          (optional, defaults to True)

    Returns:

        A: the recovered matrix.

        E: the differences between the input matrix and the recovered matrix,
          so A+E=D.
          (Note, generally E is not important, but Lin-Chen-Ma return
           it so we do the same here.)
    """
    assert len(u.shape) == 1, 'u must be one dimensional'
    assert len(v.shape) == 1, 'v must be one dimensional'
    assert len(d.shape) == 1, 'd must be one dimensional'
    assert 0 <= np.max(u) < m, 'An entry in u is invalid'
    assert 0 <= np.max(v) < n, 'An entry in v is invalid'

    if epsilon1 is None:
        # The default values for epsilon1 is from bottom of page
        # 12 in Lin-Cheyn-Ma.
        epsilon1 = 1e-7
    if epsilon2 is None:
        # The default values for epsilon2 is from bottom of page
        # 12 in Lin-Chen-Ma.
        epsilon2 = 1e-6

    # The minimum value of the observed entries of D
    minD = np.min(d)

    # We want to keep around a sparse matrix version of D, but we need to be
    # careful about 0 values in d, we don't want them to get discarded when we
    # convert to a sparse matrix! In particular, we are using the sparse matrix
    # in a slightly odd way.  We intend that D stores both 0 and non-zero
    # values, and that the entries of D which are not stored are *unknown* (and
    # not necessarily 0).  Therefore, we process the input d to make all 0
    # entries "small" numbers relative to its smallest value.
    for i in range(len(d)):
        if d[i] == 0:
            d[i] = minD * epsilon1

    # Create the required sparse matrices.  Note, u,v,d might have
    # repeats, and that is ok since the sp.coo_matrix handles
    # that case, and we don't actually use d after here.
    D = sp.csc_matrix(sp.coo_matrix((d, (u, v)), shape=[m, n]))

    # The Frobenius norm of the observed entries of D.  This is
    # just the 2-norm of the *vector* of entries.
    partialFrobeniusD = sparseFrobeniusNorm(D)

    # The SVD of the answer A
    U = np.matrix(np.zeros([m, maxRank]))
    S = np.zeros([maxRank])
    VT = np.matrix(np.zeros([maxRank, n]))

    # Compute the largest singular values of D (assuming the unobserved entries
    # are 0. I am not convinced this is principled, but I believe it it what
    # they do in the paper.
    dummy, S0, dummy = sparseSVDUpdate(D, U[:, 0], np.array([S[0]]), VT[0, :])

    if mu_0 is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.  I believe that the use the
        # spectral norm of D (the largest singular value), where
        # the unobserved entries are assumed to be 0.
        # FIXME:  I am not sure this is principled.  I mean, why is 0 special?
        # I am pretty sure that I can break this with a inproperly scaled D.
        mu_0 = 1. / S0[0]
    if rho is None:
        # The default values for mu_0 is from bottom of page
        # 12 in Lin-Chen-Ma.
        # The flatten here is important since the ord=1 norm
        # from np.linalg.norm for a matrix is max(sum(abs(x), axis=0)), which
        # is *not* what we want.
        rho_s = len(d) / (m * n)
        rho = 1.2172 + 1.8588 * rho_s

    # The sparse Lagrange multiplers
    Y_0 = D * 0.0

    # The projection of A onto Omega.  This is not required
    # but is convenient to have.
    POA_0 = D * 0.0
    POA_1 = D * 0.0

    iteration = 0
    while True:
        # Break if we use too many interations
        iteration += 1
        if iteration > maxIteration:
            break

        # This is the mathematical content of the algorithm
        ###################################################
        # The full_matrices being true is required for non-square matrices
        # We know that E_0 = POA_0 - A_0 = POA_0 - U_0*S_0*VT_0
        # So,
        # [U,S,VT] = np.linalg.svd(D-E_0+Y_0/mu_0, full_matrices=False)
        # can be rewritten as
        # [U,S,VT] = np.linalg.svd(D-(POA_0 - U_0*S_0*VT_0)+Y_0/mu_0,
        #                          full_matrices=False)
        # Combining sparse terms we get
        # [U,S,VT] = np.linalg.svd( (D-POA_0+Y_0/mu_0) + U_0*S_0*VT_0,
        #                          full_matrices=False)

        [U, S, VT] = minNucPlusFrob(D - POA_0 + Y_0 / mu_0, U, S, VT, mu_0)

        # and we compute the projection of A onto Omega
        # Note, making the temp array and then creating the sparse
        # matrix all at once is *much* faster.
        POA_1 = projSVD(U, S, VT, u, v)

        # POATmp = np.zeros([len(d)])
        # # FIXME:  Needs to be numba
        # for i in range(len(d)):
        #     POATmp[i] = U[u[i], :] * np.diag(S) * VT[:, v[i]]
        # POA_1 = sp.csc_matrix(sp.coo_matrix((POATmp, (u, v)), shape=[m, n]))

        # Update the Lagrange mutiplier
        # We have that
        # E_1 = POA_1 - A_1 = POA_1 - U_1*S_1*VT_1
        # So we can plug into
        # Y_1 = Y_0 + mu_0*(D-A_1-E_1)
        # to get
        # Y_1 = Y_0 + mu_0*(D-A_1-(POA_1 - A_1))
        # so
        # Y_1 = Y_0 + mu_0*(D-POA_1)

        Y_1 = Y_0 + mu_0 * (D - POA_1)
        ###################################################

        # If the method is converging well then increase mu_0 to focus
        # more on the constraint.  if
        # mu_0*np.linalg.norm(POA_1-POA_0,ord=2)/partialFrobeniusD <
        # epsilon2: Again, I don't know how to compute the spectral
        # norm of a partially observed matrix, so I replace with the
        # Froebenius norm on the observed entries FIXME: Attempt to
        # justify later.
        if (mu_0 * sparseFrobeniusNorm(POA_1 - POA_0) / partialFrobeniusD
                < epsilon2):
            mu_0 = rho * mu_0

        # stopping criterion from page 12 of Lin, Chen, and Ma.
        # criterion1 = np.linalg.norm(D-A_1-E_1, ord='fro')
        #   /np.linalg.norm(D, ord='fro')
        # criterion1 = np.linalg.norm(D-A_1-(POA_1 - A_1), ord='fro')
        #   /np.linalg.norm(D, ord='fro')
        # criterion1 = np.linalg.norm(D-POA_1), ord='fro')
        #   /np.linalg.norm(D, ord='fro')
        # FIXME:  I may need to justify the change from the full Froebenius
        #         norm to the partial one.
        criterion1 = sparseFrobeniusNorm(D - POA_1) / partialFrobeniusD
        # criterion2 = np.min([mu_0,np.sqrt(mu_0)])
        #   *np.linalg.norm(E_1-E_0, ord='fro')/np.linalg.norm(D, ord='fro')
        # This is the one place where I depart from Lin-Chen-Ma.  The stopping
        # criterion there have right about equation uses A and POA.  As I want
        # the algorithm to be fast I ignore the A part, since that would be
        # O(mn)
        # FIXME:  Need to justify
        # FIXME:  I may need to justify the change from the full Froebenius
        #         norm to the partial one.
        criterion2 = np.min([mu_0, np.sqrt(mu_0)]) * \
            sparseFrobeniusNorm(POA_1 - POA_0) / partialFrobeniusD

        if verbose:
            if iteration == 1:
                print("printing")
                print(("iteration criterion1 epsilon1 " +
                      "criterion2 epsilon2 rho      mu"))
            if iteration % 10 == 0:
                print(('%9d %10.2e %8.2e %10.2e %8.2e %8.2e %8.2e' %
                      (iteration, criterion1, epsilon1, criterion2, epsilon2,
                       rho, mu_0)))

        # If both error criterions are satisfied stop the algorithm
        if criterion1 < epsilon1 and criterion2 < epsilon2:
            break

        Y_0 = Y_1.copy()
        POA_0 = POA_1.copy()

    return [U, S, VT]


def test_compare():
    from dimredu.MCviaCVXPy import MC as MCCVXPy
    from dimredu.MCviaIALM import MC as MCSlow
    m = 5
    n = 7
    U = np.matrix(np.random.random(size=[m, 1]))
    V = np.matrix(np.random.random(size=[n, 1]))
    D = U * V.T
    Omega = np.zeros([m, n])
    tmp = np.random.uniform(size=[m, n])
    Omega[tmp < 0.7] = 1

    ACVXPy = MCCVXPy(np.multiply(Omega, D), Omega)
    ASlow, ESlow = MCSlow(np.multiply(Omega, D), Omega, maxIteration=200)

    u = []
    v = []
    d = []
    for i in range(m):
        for j in range(n):
            if Omega[i, j]:
                u.append(i)
                v.append(j)
                d.append(D[i, j])
    u = np.array(u)
    v = np.array(v)
    d = np.array(d)

    [U, S, VT] = MC(m, n, u, v, d, 4)
    AFast = U * np.diag(S) * VT

    assert np.allclose(ACVXPy, D, atol=1e-1)
    assert np.allclose(ASlow, D, atol=1e-1)
    assert np.allclose(AFast, D, atol=1e-1)


def profile_large():
    m = 200
    n = 500
    samples = int(0.4 * m * n)
    print(('samples', samples))
    np.random.seed(1234)
    origU = np.matrix(np.random.random(size=[m, 2]))
    origV = np.matrix(np.random.random(size=[n, 2]))

    u = []
    v = []
    d = []
    for i in range(samples):
        # Note, there may be repeats in d, but that is ok
        # since the solver very early calls the coo_matrix function,
        # and function gracefully handles repeats.
        uTmp = np.random.randint(0, m)
        vTmp = np.random.randint(0, n)
        u.append(uTmp)
        v.append(vTmp)
        d.append(float(origU[uTmp, :] * (origV.T)[:, vTmp]))

    u = np.array(u)
    v = np.array(v)
    d = np.array(d)
    # The choice of maxRank is apparently important for convergence.   Even
    # though the final answer is only rank 2, we appear to need the extra
    # dimensions to make it converge.
    maxRank = 10
    [U, S, VT] = MC(m, n, u, v, d, maxRank, rho=1.01, maxIteration=500)

    # Randomly sample the errors to see how we did
    errorSamples = 500
    errors = np.zeros([errorSamples])
    for i in range(errorSamples):
        uTmp = np.random.randint(0, m)
        vTmp = np.random.randint(0, n)
        orig = origU[uTmp, :] * (origV.T)[:, vTmp]
        computed = U[uTmp, :] * np.diag(S) * VT[:, vTmp]
        errors[i] = np.abs(orig - computed)
    print((np.max(errors)))


def profile():
    import cProfile
    cProfile.run('profile_large()', 'stats')


if __name__ == '__main__':
    test_compare()
