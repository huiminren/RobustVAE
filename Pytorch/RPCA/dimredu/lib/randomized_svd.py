#! /usr/bin/env python
"""This is taken from sklearn/utils/exmath.py and updated to work with
linear operators

FIXME:  Do the right thing with the license
"""

from scipy import linalg
import numpy as np
# from sklearn.utils.fixes import qr_economic
from sklearn.utils.validation import check_random_state
from sklearn.utils.extmath import svd_flip


def randomized_range_finder(A, AT, size, n_iter, random_state=None,
                            n_iterations=None):
    """Computes an orthonormal matrix whose range approximates the range of A.

    Args:
        A: The input data matrix
        size: Size of the return array
        n_iter: Number of power iterations used to stabilize the result
        random_state: A random number generator instance

    Returns:
        Q: 2D array
           A (size x size) projection matrix, the range of which
           approximates well the range of the input matrix A.

    Notes:
        Follows Algorithm 4.3 of Finding structure with randomness:
        Stochastic algorithms for constructing approximate matrix
        decompositions Halko, et al., 2009 (arXiv:909)
        http://arxiv.org/pdf/0909.4061
    """
    if n_iterations is not None:
        warnings.warn("n_iterations was renamed to n_iter for consistency "
                      "and will be removed in 0.16.", DeprecationWarning)
        n_iter = n_iterations
    random_state = check_random_state(random_state)

    # generating random gaussian vectors r with shape: (A.shape[1], size)
    R = random_state.normal(size=(A.shape[1], size))

    # sampling the range of A using by linear projection of r
    # Y = safe_sparse_dot(A, R)
    Y = A * R
    del R

    # perform power iterations with Y to further 'imprint' the top
    # singular vectors of A in Y
    for i in range(n_iter):
        # Y = safe_sparse_dot(A, safe_sparse_dot(A.T, Y))
        Y = A * (AT * Y)

    # extracting an orthonormal basis of the A range samples
    # Q, R = qr_economic(Y)
    Q = np.array(np.linalg.qr(Y, mode='reduced')[0])

    return Q


def randomized_svd(M, MT, n_components, n_oversamples=10, n_iter=0,
                   transpose='auto', flip_sign=True, random_state=0,
                   n_iterations=None):
    """Computes a truncated randomized SVD

    Args:
      M: ndarray or sparse matrix
          Matrix to decompose

      n_components: int
          Number of singular values and vectors to extract.

      n_oversamples: int (default is 10)
          Additional number of random vectors to sample the range of M so as
          to ensure proper conditioning. The total number of random vectors
          used to find the range of M is n_components + n_oversamples.

      n_iter: int (default is 0)
          Number of power iterations (can be used to deal with very noisy
          problems).

      transpose: True, False or 'auto' (default)
          Whether the algorithm should be applied to M.T instead of M. The
          result should approximately be the same. The 'auto' mode will
          trigger the transposition if M.shape[1] > M.shape[0] since this
          implementation of randomized SVD tend to be a little faster in that
          case).

      flip_sign: boolean, (True by default)
          The output of a singular value decomposition is only unique up to a
          permutation of the signs of the singular vectors. If `flip_sign` is
          set to `True`, the sign ambiguity is resolved by making the largest
          loadings for each component in the left singular vectors positive.

      random_state: RandomState or an int seed (0 by default)
          A random number generator instance to make behavior

    Notes:
      This algorithm finds a (usually very good) approximate truncated
      singular value decomposition using randomization to speed up the
      computations. It is particularly fast on large matrices on which
      you wish to extract only a small number of components.

    References:
      * Finding structure with randomness: Stochastic algorithms for constructing
        approximate matrix decompositions
        Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

      * A randomized algorithm for the decomposition of matrices
        Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    """
    if n_iterations is not None:
        warnings.warn("n_iterations was renamed to n_iter for consistency "
                      "and will be removed in 0.16.", DeprecationWarning)
        n_iter = n_iterations

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if transpose == 'auto' and n_samples > n_features:
        transpose = True
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        tmp = M
        M = MT
        MT = tmp

    Q = randomized_range_finder(M, MT, n_random, n_iter, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    #B = safe_sparse_dot(Q.T, M)
    B = (MT * Q).T

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        U, V = svd_flip(U, V)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]
