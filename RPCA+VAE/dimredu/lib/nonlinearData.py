#! /usr/bin/env python
"""Some non-linear data sets for testing and visualization.
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d


def swissRoll(u, v, f=None):
    assert u.shape[0] == v.shape[0], 'u and v must have same length'
    assert len(u.shape) == 1, 'u must be an array'
    assert len(v.shape) == 1, 'v must be an array'

    x = []
    y = []
    z = []
    uN = []
    vN = []
    for i in range(len(u)):
        if (f is None) or (f(u[i], v[i]) < 0.5):
            uN.append(u[i])
            vN.append(v[i])
            x.append((u[i] + 1) * np.cos(2 * np.pi * u[i]))
            y.append((u[i] + 1) * np.sin(2 * np.pi * u[i]))
            z.append(-v[i] * np.pi)
    return [np.array(uN), np.array(vN),
            np.array(x), np.array(y), np.array(z)]


def WPISwissRoll(u, v, filename=None):
    if filename is None:
        import playground
        import os
        directory = os.path.dirname(dimredu.__file__)
        filename = os.path.join(directory, 'src/test/wpi.png')
    mask = ndimage.imread(filename)[:, :, 3] / 255.
    um = np.linspace(0, 1, mask.shape[0])
    vm = np.linspace(0, 1, mask.shape[1])
    # There is a wierd ordering of the arguments for this function.
    f = interp2d(vm, um, mask)

    return swissRoll(u, v, f)

def check1():
    import mayavi.mlab as ml
    size = 5000
    u = np.random.random(size=[size])
    v = np.random.random(size=[size])

    uN, vN, x, y, z = WPISwissRoll(u, v)

    ml.figure(figure=ml.gcf(), size=(2000, 2000))
    ml.clf()
    ml.points3d(x, y, z, uN, mode='sphere',
                scale_factor=0.1, scale_mode='none')
    ml.show()


def check2():
    import matplotlib.pylab as py
    from mpl_toolkits.mplot3d import Axes3D
    size = 10000
    u = np.random.random(size=[size])
    v = np.random.random(size=[size])

    uN, vN, x, y, z = WPISwissRoll(u, v)

    fig = py.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    py.show()


if __name__ == '__main__':
    check2()
