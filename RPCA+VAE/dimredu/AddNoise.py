from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def generateNoisyGraph():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    np.random.seed (595)
    noise_factor = 0.2
    x_train_noisy = x_train + noise_factor * np.random.normal (loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal (loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train, x_test, x_train_noisy, x_test_noisy

def plotGraph(graph, n = 10):
    plt.figure (figsize=(20, 2))
    for i in range (n):
        ax = plt.subplot (1, n, i + 1)
        plt.imshow (graph[i].reshape (28, 28))
        plt.gray ()
        ax.get_xaxis ().set_visible (False)
        ax.get_yaxis ().set_visible (False)
    plt.show ()
