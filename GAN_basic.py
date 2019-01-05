#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angela Y
Reference:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py

"""
# TODO double check nois_dim
# TODO discuss total loss
# TODO discuss plot1
# TODO huimin double check plot reshape to 64, 64


from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import time
import os


def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i, min(l, i + n))


class GAN_basic(object):
    def __init__(self, sess, noise_dim=5, learning_rate=1e-3, batch_size=64):
        """
        sess: tf.Session()
        noise_dim: noise dimension (like dimension in latent layer)
        input_dim: input dimension
        dec_in_channels: dimension in channel
        learning_rate: learning rate for loss
        """

        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size


        # Network Params
        image_dim = 784  # 28*28 pixels
        gen_hidden_dim1 = 256
        gen_hidden_dim2 = 384
        gen_hidden_dim3 = 512

        disc_hidden_dim1 = 512
        disc_hidden_dim2 = 384
        disc_hidden_dim3 = 256
        noise_dim = 5  # Noise data points
        # Store layers weight & bias
        self.weights = {
            'gen_hidden1': tf.Variable(self.glorot_init([noise_dim, gen_hidden_dim1])),
            'gen_hidden2': tf.Variable(self.glorot_init([gen_hidden_dim1, gen_hidden_dim2])),
            'gen_hidden3': tf.Variable(self.glorot_init([gen_hidden_dim2, gen_hidden_dim3])),
            'gen_out': tf.Variable(self.glorot_init([gen_hidden_dim3, image_dim])),
            'disc_hidden1': tf.Variable(self.glorot_init([image_dim, disc_hidden_dim1])),
            'disc_hidden2': tf.Variable(self.glorot_init([disc_hidden_dim1, disc_hidden_dim2])),
            'disc_hidden3': tf.Variable(self.glorot_init([disc_hidden_dim2, disc_hidden_dim3])),
            'disc_hidden4': tf.Variable(self.glorot_init([disc_hidden_dim3, noise_dim])),
            'disc_out': tf.Variable(self.glorot_init([noise_dim, 1])),
        }

        self.biases = {
            'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim1])),
            'gen_hidden2': tf.Variable(tf.zeros([gen_hidden_dim2])),
            'gen_hidden3': tf.Variable(tf.zeros([gen_hidden_dim3])),
            'gen_out': tf.Variable(tf.zeros([image_dim])),
            'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim1])),
            'disc_hidden2': tf.Variable(tf.zeros([disc_hidden_dim2])),
            'disc_hidden3': tf.Variable(tf.zeros([disc_hidden_dim3])),
            'disc_hidden4': tf.Variable(tf.zeros([noise_dim])),
            'disc_out': tf.Variable(tf.zeros([1])),
        }

        self.build()

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())


    # A custom initialization (see Xavier Glorot init)
    def glorot_init(self, shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    def build(self):

        # Build Networks
        # Network Inputs
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='input_noise')
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name='disc_input')
        self.isTrain = tf.placeholder(dtype=tf.bool)

        # Build Generator Network
        self.G_z = self.generator(self.z, self.isTrain)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = self.discriminator(self.x, self.isTrain)
        disc_fake = self.discriminator(self.G_z, self.isTrain)

        # Build Loss
        self.G_loss = -tf.reduce_mean(tf.log(disc_fake))
        self.D_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
        # Build Optimizers
        self.G_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss)
        self.D_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss)

    # G(z)
    def generator(self, x, isTrain=True):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            hidden_layer1 = tf.matmul(x, self.weights['gen_hidden1'])
            hidden_layer1 = tf.add(hidden_layer1, self.biases['gen_hidden1'])
            hidden_layer1 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer1, training=isTrain))

            hidden_layer2 = tf.matmul(hidden_layer1, self.weights['gen_hidden2'])
            hidden_layer2 = tf.add(hidden_layer2, self.biases['gen_hidden2'])
            hidden_layer2 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer2, training=isTrain))

            hidden_layer3 = tf.matmul(hidden_layer2, self.weights['gen_hidden3'])
            hidden_layer3 = tf.add(hidden_layer3, self.biases['gen_hidden3'])
            hidden_layer3 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer3, training=isTrain))

            out_layer = tf.matmul(hidden_layer3, self.weights['gen_out'])
            out_layer = tf.add(out_layer, self.biases['gen_out'])
            out_layer = tf.nn.sigmoid(tf.layers.batch_normalization(out_layer, training=isTrain))
            return out_layer



    # D(x)
    def discriminator(self, x, isTrain=True):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            hidden_layer1 = tf.matmul(x, self.weights['disc_hidden1'])
            hidden_layer1 = tf.add(hidden_layer1, self.biases['disc_hidden1'])
            hidden_layer1 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer1, training=isTrain))

            hidden_layer2 = tf.matmul(hidden_layer1, self.weights['disc_hidden2'])
            hidden_layer2 = tf.add(hidden_layer2, self.biases['disc_hidden2'])
            hidden_layer2 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer2, training=isTrain))

            hidden_layer3 = tf.matmul(hidden_layer2, self.weights['disc_hidden3'])
            hidden_layer3 = tf.add(hidden_layer3, self.biases['disc_hidden3'])
            hidden_layer3 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer3, training=isTrain))

            hidden_layer4 = tf.matmul(hidden_layer3, self.weights['disc_hidden4'])
            hidden_layer4 = tf.add(hidden_layer4, self.biases['disc_hidden4'])
            hidden_layer4 = tf.nn.sigmoid(tf.layers.batch_normalization(hidden_layer4, training=isTrain))

            out_layer = tf.matmul(hidden_layer4, self.weights['disc_out'])
            out_layer = tf.add(out_layer, self.biases['disc_out'])
            out_layer = tf.nn.sigmoid(out_layer)
            return out_layer

    '''
    def fit(self, X_in, path, num_gen=100, num_epoch=30):
        """
        X_in: image without label
        """
        gl_loss = []
        dl_loss = []

        for epoch in range(num_epoch):
            G_losses = []
            D_losses = []

            for iter in range(X_in.shape[0] // self.batch_size):
                # update discriminator
                x_ = X_in[iter * self.batch_size:(iter + 1) * self.batch_size]
                x_.shape = (x_.shape[0], 784)

                z_ = np.random.normal(0, 1, (self.batch_size, self.noise_dim))

                loss_d_, _ = self.sess.run([self.D_loss, self.D_optim], {self.x: x_, self.z: z_, self.isTrain: True})
                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                loss_g_, _ = self.sess.run([self.G_loss, self.G_optim], {self.z: z_})
                G_losses.append(loss_g_)

            if epoch % 1 == 0:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (
                epoch, np.mean(G_losses), np.mean(D_losses)))
                self.plot(path=path, fig_name="generator_" + str(epoch) + ".png", num_gen=num_gen)

            gl_loss.append(np.mean(G_losses))
            dl_loss.append(np.mean(D_losses))

        np.save(path + "gl_loss.npy", np.array(gl_loss))
        np.save(path + "dl_loss.npy", np.array(dl_loss))
    '''


    def fit(self, X_in, path, num_gen=100, num_epoch=100):
        """
        X_in: image without label
        """
        sample_size = X_in.shape[0]
        gl_loss = []
        dl_loss = []

        for epoch in range(num_epoch):
            G_losses = []
            D_losses = []
            for one_batch in batches(sample_size, self.batch_size):
                # update discriminator
                z_ = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                loss_d_, _ = self.sess.run([self.D_loss, self.D_optim], {self.x: X_in[one_batch],
                                                                         self.z: z_, self.isTrain: True})
                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                loss_g_, _ = self.sess.run([self.G_loss, self.G_optim],
                                           {self.z: z_, self.x: X_in[one_batch], self.isTrain: True})
                G_losses.append(loss_g_)

            if epoch % 1 == 0:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (
                epoch, np.mean(G_losses), np.mean(D_losses)))
                self.plot(path=path, fig_name="generator_" + str(epoch) + ".png", num_gen=num_gen)

            gl_loss.append(np.mean(G_losses))
            dl_loss.append(np.mean(D_losses))

        np.save(path + "gl_loss.npy", np.array(gl_loss))
        np.save(path + "dl_loss.npy", np.array(dl_loss))

    def plot(self, path, fig_name, num_gen=100):

        np.random.seed(595)
        h = w = 28
        z = np.random.normal(0, 1, (num_gen, self.noise_dim))
        g = self.get_generation(z)

        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h * n, w * n))
        for i in range(n):
            for j in range(n):
                I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = g[i * n + j, :].reshape(28, 28)

        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path + fig_name)
        # plt.show()

    def get_generation(self, z):
        return self.sess.run(self.G_z, feed_dict={self.z: z, self.isTrain: False})


def main(noise_factors, debug=True):
    start_time = time.time()

    tf.set_random_seed(595)
    tf.reset_default_graph()
    # open session and initialize all variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
    # x_train = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
    x_train = mnist.train.images

    hidden_dim = 5

    batch_size = 200  # X_in.shape[0] % batch_size == 0
    num_epoch = 30
    num_gen = 100
    if debug:
        x_train = x_train[:256]  # it must be n%batch_size == 0
        batch_size = 32
        num_epoch = 2
        num_gen = 10

    for noise_factor in noise_factors:
        print("noise factor: ", noise_factor)
        path = "save_images_GAN/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + str(noise_factor) + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        gan = GAN_basic(sess, noise_dim=hidden_dim, learning_rate=1e-3, batch_size=batch_size)
        x_train_noisy.shape = (x_train_noisy.shape[0], 784)
        gan.fit(x_train_noisy, path=path, num_gen=num_gen, num_epoch=num_epoch)
    sess.close()
    print("running time: ", time.time() - start_time)


if __name__ == "__main__":
    noise_factors = np.array([0.2, 0.4])
    main(noise_factors=noise_factors, debug=False)