import time
import numpy as np
from numba import jit


def shrink(epsilon, x):
    """The shrinkage operator.
    This implementation is intentionally slow but transparent as
    to the mathematics.

    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    # try explicitly stating type of x

    output = np.array(x * 0.)
    if np.isscalar(epsilon):
        epsilon = np.ones(x.shape) * epsilon
    return jitShrink(epsilon, x, output)


@jit(nopython=True, cache=True)
def jitShrink(epsilon, x, output):
    for i in range(len(x)):
        if x[i] > epsilon[i]:
            output[i] = x[i] - epsilon[i]
        elif x[i] < -epsilon[i]:
            output[i] = x[i] + epsilon[i]
        else:
            output[i] = 0
    return output


def test_shrink():
    import numpy as np
    x = np.array([-2, -1, -0.499, 0, 0.499, 1.0, 1.5, 2])
    print(x)
    print((shrink(0.5, x)))


def test_shrink2():
    import numpy as np
    x = np.array([1, 2, 3])
    print(x)
    print((shrink([1., 1.5, 1.], x)))


if __name__ == '__main__':
    test_shrink()
    test_shrink2()
