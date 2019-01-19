#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:23:50 2018
@author: huiminren
Reference: 
    https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py
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
    def __init__(self, sess, noise_dim = 100, learning_rate = 1e-3, batch_size = 64):
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
        
        self.build()
        
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
    
    def build(self):
        # variables : input
        self.x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
        self.z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
        self.isTrain = tf.placeholder(dtype=tf.bool)
        
        # networks : generator
        self.G_z = self.generator(self.z, self.isTrain)
        
        # networks : discriminator
        D_real, D_real_logits = self.discriminator(self.x, self.isTrain)
        D_fake, D_fake_logits = self.discriminator(self.G_z, self.isTrain, reuse=True)
        
        # loss for each network
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, 
                                                                             labels=tf.ones([self.batch_size, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, 
                                                                             labels=tf.zeros([self.batch_size, 1, 1, 1])))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, 
                                                                             labels=tf.ones([self.batch_size, 1, 1, 1])))
        
        # trainable variables for each network
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        G_vars = [var for var in T_vars if var.name.startswith('generator')]
        
        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            self.G_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.G_loss, var_list=G_vars)
        

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    # G(z)
    def generator(self, x, isTrain=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
        
            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
            lrelu1 = self.lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
        
            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        
            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        
            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
            # output layer
            conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv5)
        
            return o
        
    # D(x)
    def discriminator(self, x, isTrain=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # 1st hidden layer
            conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu1 = self.lrelu(conv1, 0.2)
        
            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        
            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        
            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
            # output layer
            conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
            o = tf.nn.sigmoid(conv5)
        
            return o, conv5

    
    def fit(self, X_in, path, num_gen = 100, num_epoch =100):
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
                z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, self.noise_dim))
                loss_d_, _ = self.sess.run([self.D_loss, self.D_optim], {self.x: X_in[one_batch], 
                                           self.z: z_, self.isTrain: True})
                D_losses.append(loss_d_)
        
                # update generator
                # why set a new z_ to update generator?
                z_ = np.random.normal(0, 1, (self.batch_size, 1, 1, self.noise_dim))
                loss_g_, _ = self.sess.run([self.G_loss, self.G_optim], {self.z: z_, self.x: X_in[one_batch], self.isTrain: True})
                G_losses.append(loss_g_)
                
            if epoch % 1 == 0:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (epoch, np.mean(G_losses), np.mean(D_losses)))
                self.plot(path = path, fig_name = "generator_"+str(epoch)+".png", num_gen = num_gen)
            
            gl_loss.append(np.mean(G_losses))
            dl_loss.append(np.mean(D_losses))
                
        np.save(path+"gl_loss.npy",np.array(gl_loss))
        np.save(path+"dl_loss.npy",np.array(dl_loss))
                    
    def plot(self, path, fig_name, num_gen = 100):
        
        np.random.seed(595)
        h = w = 64
        z = np.random.normal(0, 1, (num_gen, 1, 1, self.noise_dim))
        g = self.get_generation(z)
        
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = g[i*n+j, :].reshape(64, 64)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
        plt.show()
        
        
        
    def get_generation(self, z):
        return self.sess.run(self.G_z, feed_dict={self.z: z, self.isTrain: False})
    

def main(noise_factors,debug = True):
    start_time = time.time()
    
    tf.reset_default_graph()
    # open session and initialize all variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
    x_train = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
#    x_train = (x_train - 0.5) / 0.5  # normalization; range: -1 ~ 1
    
    batch_size = 200 # X_in.shape[0] % batch_size == 0
    num_epoch = 30
    num_gen = 100
    if debug:
        x_train = x_train[:256] # it must be n%batch_size == 0
        batch_size = 32
        num_epoch = 2
        num_gen = 10
    
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "save_images_DCGAN_z/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        tf.reset_default_graph()
        dcgan = DCGAN(sess, noise_dim = 100, learning_rate = 1e-3, batch_size = batch_size)
        dcgan.fit(x_train_noisy,path = path, num_gen = num_gen, num_epoch = num_epoch)
    sess.close()
    print("running time: ",time.time()-start_time)
    
if __name__ == "__main__":  
    noise_factors = np.array([0.2,0.4])
    main(noise_factors = noise_factors,debug = False)