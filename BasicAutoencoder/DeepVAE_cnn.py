#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:02:26 2018

@author: huiminren

Variational Auto-Encoder Example.
Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.
References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

fork from form https://github.com/FelixMohr/Deep-learning-with-Python/blob/master/VAE.ipynb 
"""
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
        yield range(i,min(l,i+n))
        
class Deep_CNNVAE(object):
    def __init__(self, sess, learning_rate = 1e-3, n_latent = 8, 
                 input_dim = 28, dec_in_channels = 1):
        
        self.learning_rate = learning_rate
        self.n_latent = n_latent
        self.input_dim = input_dim
        self.dec_in_channels = dec_in_channels
        self.reshaped_dim = [-1, 7, 7, self.dec_in_channels]
        self.inputs_decoder = int(49*self.dec_in_channels/2)
        
        self.build()
        
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        
    # Build the network and loss functions
    def build(self):
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim, self.input_dim], name = 'X')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name = 'keep_prob')
        
        self.sampled, self.mn, self.sd = self.encoder(self.X_in, self.keep_prob)
        self.dec = self.decoder(self.sampled, self.keep_prob)
        
        unreshaped = tf.reshape(self.dec, [-1, self.input_dim*self.input_dim])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, 
                                      tf.reshape(self.X_in,shape=[-1,self.input_dim*self.input_dim])), 1)
        self.img_loss = tf.reduce_mean(img_loss)
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
        # latent_loss = -0.5 * tf.reduce_sum(sd - tf.square(mn) - tf.exp(sd), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        return
    
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    def encoder(self, X_in, keep_prob):
        activation = self.lrelu
        with tf.variable_scope("encoder", reuse=None):
            X = tf.reshape(X_in, shape=[-1, self.input_dim, self.input_dim, self.dec_in_channels])
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=self.n_latent)
            sd = 0.5 * tf.layers.dense(x, units=self.n_latent)            
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd))
            
            return z, mn, sd   
        
    def decoder(self, sampled_z, keep_prob):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=self.inputs_decoder, activation=self.lrelu)
            x = tf.layers.dense(x, units=self.inputs_decoder * 2 + 1, activation=self.lrelu)
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
            
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.input_dim*self.input_dim, activation=tf.nn.sigmoid)
            img = tf.reshape(x, shape=[-1, self.input_dim, self.input_dim])
            
            return img        
        
    def run_single_step(self, X_in,keep_prob):
        _, loss, img_loss, latent_loss = self.sess.run(
        [self.optimizer, self.loss, self.img_loss, self.latent_loss],
        feed_dict={self.X_in: X_in, self.keep_prob:keep_prob}
        )
        return loss, img_loss, latent_loss
    
    def fit(self, X_in, path = "", file_name="", num_epoch = 100, batch_size = 64, keep_prob = 1.0):
        ls_loss = []
        sample_size = X_in.shape[0]
        for epoch in range(num_epoch):
            for one_batch in batches(sample_size, batch_size):
                loss, img_loss, latent_loss = self.run_single_step(X_in[one_batch],keep_prob)
            ls_loss.append(loss)
            if epoch % 1 == 0:
                print('[epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                    epoch, loss, img_loss, latent_loss))
        np.save(path+file_name,np.array(ls_loss))
                
    def plot(self,FLAG_gen = True, x="", num_gen=10, path="", fig_name=""):
        """
        FLAG_gen: flag of generation or reconstruction. True = generation
        x: reconstruction input
        num_gen: number of generation
        path: path of saving
        fig_name: name of saving
        """
        
        if not os.path.exists(path):
            os.mkdir(path)
            
        np.random.seed(595)
        h = w = 28
        
        if FLAG_gen:
            z = np.random.normal(size=[num_gen, self.n_latent])
            rvae_generated = self.generator(z,1) # get generation
        else:
            rvae_generated = self.reconstructor(x,1)
        
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = rvae_generated[i*n+j, :].reshape(28, 28)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
        plt.show()
        
    # x -> x_hat
    def reconstructor(self, x, keep_prob):
        x_hat = self.sess.run(self.dec, feed_dict={self.X_in: x, self.keep_prob:keep_prob})
        return x_hat

    # z -> x_hat
    def generator(self, z, keep_prob):
        x_hat = self.sess.run(self.dec, feed_dict={self.sampled: z, self.keep_prob:keep_prob})
        return x_hat
    
    # x -> z
    def transformer(self, x, keep_prob):
        z = self.sess.run(self.sampled, feed_dict={self.X_in: x, self.keep_prob:keep_prob})
        return z    

if __name__ == "__main__":
    start_time = time.time()
    
    root = 'save_images/'
    if not os.path.isdir(root):
        os.mkdir(root)
    
    batch_size = 64
    n_latent = 8
    input_dim = 28
    
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    X_in = mnist.train.images
    X_in = X_in.reshape(-1,input_dim,input_dim)
    print("X_in",X_in[1,2,3])
    
    test = mnist.test.next_batch(100)[0]
    test = test.reshape(-1,input_dim,input_dim)
    
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        cnnvae = Deep_CNNVAE(sess = sess, learning_rate = 1e-3, n_latent = n_latent,
                             input_dim = input_dim, dec_in_channels = 1)
        cnnvae.fit(X_in, path = "", file_name="", num_epoch = 1, batch_size = batch_size,keep_prob = 1.0)
        
#        saver.restore(sess,"model_cnn.ckpt")
        # Test the trained model: reconstruction
        x_reconstructed = cnnvae.reconstructor(test,1)
        x_transformer = cnnvae.transformer(test,1)
        
        np.save(root+"rvae_recon.npy",x_reconstructed)
        np.save(root+"rvae_transform.npy",x_transformer)
        
        
        w = h = input_dim
        
        n = np.sqrt(batch_size).astype(np.int32)
        I_reconstructed = np.empty((h*n, 2*w*n))
        for i in range(n):
            for j in range(n):
                x = np.concatenate(
                    (x_reconstructed[i*n+j, :].reshape(h, w), 
                     test[i*n+j, :].reshape(h, w)),
                    axis=1
                )
                I_reconstructed[i*h:(i+1)*h, j*2*w:(j+1)*2*w] = x
        
        plt.figure(figsize=(10, 20))
        plt.title('reconstruction of test images')
        plt.imshow(I_reconstructed, cmap='gray')
        plt.show()
        plt.savefig(root+"cnnvae_reconstruction.png")

        # Test the trained model: generation
        # Sample noise vectors from N(0, 1)
        z = np.random.normal(size=[batch_size, n_latent])
        x_generated = cnnvae.generator(z,1)
        np.save(root+"rvae_generated.npy",x_generated)
        
        n = np.sqrt(batch_size).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(input_dim, input_dim)
        
        plt.figure(figsize=(8, 8))
        plt.title('generation from random noise')
        plt.imshow(I_generated, cmap='gray')
        plt.show()
        plt.savefig(root+"cnnvae_generator.png")
