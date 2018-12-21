#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:23:50 2018
@author: huiminren
Reference: 
    https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dcgan.py
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import time
import os


def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))
        
class DCGAN(object):
    def __init__(self, sess, noise_dim = 100, learning_rate = 1e-3):
        """
        sess: tf.Session()
        noise_dim: noise dimension (like dimension in latent layer)
        dec_in_channels: dimension in channel
        learning_rate: learning rate for loss
        """
        
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        
        self.build()
        
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
    
    def build(self):
        # Build Networks
        # Network Inputs
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        
        # Build Generator Network
        self.gen_sample = self.generator(self.noise_input)
        
        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = self.discriminator(self.real_image_input)
        disc_fake = self.discriminator(self.gen_sample, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)
        
        # Build the stacked generator/discriminator
        stacked_gan = self.discriminator(self.gen_sample, reuse=True)

        # Build Targets (real or fake images)
        self.disc_target = tf.placeholder(tf.int32, shape=[None])
        self.gen_target = tf.placeholder(tf.int32, shape=[None])
        
        # Build Loss
        self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_concat, labels=self.disc_target))
        self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_gan, labels=self.gen_target))
        
        # Build Optimizers
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator Network Variables
        self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        
        # Create training operations
        self.train_gen = self.optimizer_gen.minimize(self.gen_loss, var_list=self.gen_vars)
        self.train_disc = self.optimizer_disc.minimize(self.disc_loss, var_list=self.disc_vars)
        

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    """
    https://github.com/vdumoulin/conv_arithmetic
    expalanation of conv2d and conv2d_transpose
    """
    # Generator Network
    def generator(self, X_in, reuse = False):
        """ 
        Input: X_in: Noise; reuse: Boolean 
        Output: Image
        """
        with tf.variable_scope('Generator', reuse=reuse):
            # TensorFlow Layers automatically create variables and calculate their
            # shape, based on the input.
            x = tf.layers.dense(X_in, units=6 * 6 * 128)
            x = tf.nn.tanh(x)
            # Reshape to a 4-D array of images: (batch, height, width, channels)
            # New shape: (batch, 6, 6, 128)
            x = tf.reshape(x, shape=[-1, 6, 6, 128])
            # Deconvolution, image shape: (batch, 14, 14, 64)
            x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
            # Deconvolution, image shape: (batch, 28, 28, 1)
            x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
            # Apply sigmoid to clip values between 0 and 1
            x = tf.nn.sigmoid(x)
        return x
    
    # Discriminator Network
    def discriminator(self, X_in, reuse = False):
        """ 
        Input: X_in: Image; reuse: boolean 
        Output: Prediction Real/Fake Image
        """
        with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
            x = tf.layers.conv2d(X_in, 64, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.layers.conv2d(x, 128, 5)
            x = tf.nn.tanh(x)
            x = tf.layers.average_pooling2d(x, 2, 2)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 1024)
            x = tf.nn.tanh(x)
            # Output 2 classes: Real and Fake images
            x = tf.layers.dense(x, 2)
        return x
    
    
    def fit(self, X_in, path, num_gen = 100, num_epoch =100, batch_size = 64):
        """
        X_in: image without label
        """
        sample_size = X_in.shape[0]
        X_in = np.reshape(X_in,[-1,28,28,1])
        gl_loss = []
        dl_loss = []
        
        for epoch in range(num_epoch):
            tmp_gl = []
            tmp_dl = []
            for one_batch in batches(sample_size, batch_size):
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, self.noise_dim])
                # Prepare Targets (Real image: 1, Fake image: 0)
                # The first half of data fed to the generator are real images,
                # the other half are fake images (coming from the generator).
                batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
                # Generator tries to fool the discriminator, thus targets are 1.
                batch_gen_y = np.ones([batch_size])
                ######## why set inside of batch ????
                # Training
                feed_dict = {self.real_image_input: X_in[one_batch], self.noise_input: z,
                             self.disc_target: batch_disc_y, self.gen_target: batch_gen_y}
                _, _, gl, dl = self.sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                feed_dict=feed_dict)
                tmp_gl.append(gl)
                tmp_dl.append(dl)
            gl_loss.append(np.mean(tmp_gl))
            dl_loss.append(np.mean(tmp_dl))
            if epoch % 1 == 0:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (epoch, np.mean(tmp_gl), np.mean(tmp_dl)))
                self.plot(path = path, fig_name = "generator_"+str(epoch)+".png", num_gen = num_gen)
                
        np.save(path+"gl_loss.npy",np.array(gl_loss))
        np.save(path+"dl_loss.npy",np.array(dl_loss))
                    
    def plot(self, path, fig_name, num_gen = 100):
        
        np.random.seed(595)
        h = w = 28
        z = np.random.uniform(-1., 1., size=[num_gen, self.noise_dim])
        g = self.get_generation(z)
        
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = g[i*n+j, :].reshape(28, 28)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
        plt.show()
        
    def get_generation(self, z):
        return self.sess.run(self.gen_sample, feed_dict={self.noise_input: z})
    

def main(noise_factors,debug = True):
    start_time = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train = mnist.train.images
    
    batch_size = 256
    num_epoch = 30
    num_gen = 100
    if debug:
        x_train = x_train[:100]
        batch_size = 32
        num_epoch = 2
        num_gen = 10
    
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "save_images_DCGAN/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        tf.reset_default_graph()
        sess = tf.Session()
        dcgan = DCGAN(sess, noise_dim = 100, learning_rate = 1e-3)
        dcgan.fit(x_train_noisy,path = path, num_gen = num_gen, num_epoch = num_epoch, batch_size = batch_size)
        sess.close()
    
    print("running time: ",time.time()-start_time)
    
if __name__ == "__main__":  
    noise_factors = np.array([0.2])
    main(noise_factors = noise_factors,debug = True)