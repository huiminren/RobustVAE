#!/usr/bin/env python3
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
plt.switch_backend('agg')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import time
import os
import sys

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))
        
def corrupt(X_in,corNum=10):
    X = X_in.copy()
    dimension = X.shape[1]
    X = X.reshape(-1,dimension*dimension)
    N,p = X.shape[0],X.shape[1]
    for i in range(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j] > 0.5:
                X[i,j] = 0
            else:
                X[i,j] = 1
    X = X.reshape(-1,dimension,dimension,1)
    return X

class Deep_CNNVAE(object):
    def __init__(self, sess, learning_rate = 1e-3, n_latent = 100):
        
        self.learning_rate = learning_rate
        self.n_latent = n_latent
        
        self.build()
        
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        
    # Build the network and loss functions
    def build(self):
        self.X_in = tf.placeholder(dtype=tf.float32, shape=(None, 64,64,1), name = 'X')
        self.isTrain = tf.placeholder(dtype=tf.bool)  
        
        self.sampled, self.mn, self.sd = self.encoder(self.X_in, self.isTrain)
        self.dec = self.decoder(self.sampled, self.isTrain)
        
        unreshaped = tf.reshape(self.dec, [-1, 64*64])
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, 
                                      tf.reshape(self.X_in,shape=[-1,64*64])), 1)
        self.img_loss = tf.reduce_mean(img_loss)
        # the following formula is from author FelixMohr
#        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mn) - tf.exp(2.0 * self.sd), 1)
        # the following formula is the same as VAE
        latent_loss = -0.5 * tf.reduce_sum(1.0 + self.sd - tf.square(self.mn) - tf.exp(self.sd), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)
        self.loss = tf.reduce_mean(self.img_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        return
    
    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
    
    def encoder(self, X_in, isTrain=True):
        with tf.variable_scope("encoder", reuse=None):
            print(X_in)
            conv1 = tf.layers.conv2d(X_in, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu1 = self.lrelu(conv1, 0.2)
            print(lrelu1)
            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
            print(lrelu2)
            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
            print(lrelu3)
            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
            print(lrelu4)
            # output layer
            conv5 = tf.layers.conv2d(lrelu4, 100, [4, 4], strides=(1, 1), padding='valid')
            print(conv5)
            
            x = tf.contrib.layers.flatten(conv5)
            print(x)
            mn = tf.layers.dense(x, units=self.n_latent)
            sd = tf.layers.dense(x, units=self.n_latent)            
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent])) 
            z  = mn + tf.multiply(epsilon, tf.exp(sd))
            print(z)
            return z, mn, sd   
        
    def decoder(self, sampled_z,isTrain=True):
        with tf.variable_scope("decoder", reuse=None):
                        # 1st hidden layer
            print("sample z",sampled_z)
            sampled_z = tf.reshape(sampled_z,shape=[-1,1,1,100])
            
            conv1 = tf.layers.conv2d_transpose(sampled_z, 1024, [4, 4], strides=(1, 1), padding='valid')
            lrelu1 = self.lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
            print(lrelu1)
            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
            print(lrelu2)
            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
            print(lrelu3)
            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
            print(lrelu4)
            # output layer
            conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
            img = tf.nn.tanh(conv5)
            print(conv5)
            
            return img        
        
    def run_single_step(self, X_in):
        _, loss, img_loss, latent_loss = self.sess.run(
        [self.optimizer, self.loss, self.img_loss, self.latent_loss],feed_dict={self.X_in: X_in, self.isTrain: True}
        )
        return loss, img_loss, latent_loss
    
    def fit(self, X_in, path = "", file_name="", num_epoch = 100, batch_size = 64):
        ls_loss = []
        sample_size = X_in.shape[0]
        for epoch in range(num_epoch):
            for one_batch in batches(sample_size, batch_size):
                loss, img_loss, latent_loss = self.run_single_step(X_in[one_batch])
            ls_loss.append(loss)
            if epoch % 100 == 0:
                print('[epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                    epoch, loss, img_loss, latent_loss))
#                self.gen_plot(FLAG_gen = True, x="", num_gen=100, path=path, fig_name="generator_"+str(epoch)+".png")
#            if epoch == num_epoch-1:
#                self.gen_plot(FLAG_gen = True, x="", num_gen=100, path=path, fig_name="generator_"+str(epoch)+".png")
#                print("gid geneartion:")
#                self.generation_fid(path = path)
                
        np.save(path+file_name,np.array(ls_loss))
                
    def gen_plot(self,FLAG_gen = True, x="", num_gen=10, path="", fig_name=""):
        """
        FLAG_gen: flag of generation or reconstruction. 
            True = generation False = Reconstruction
        x: reconstruction input
        num_gen: number of generation
        path: path of saving
        fig_name: name of saving
        """
        
        if not os.path.exists(path):
            os.mkdir(path)
            
        np.random.seed(595)
        h = w = 64
        
        if FLAG_gen:
            z = np.random.normal(size=[num_gen, self.n_latent])
            rvae_generated = self.generator(z) # get generation
        else:
            rvae_generated = self.reconstructor(x)
        
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = rvae_generated[i*n+j, :].reshape(64, 64)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
        plt.show()
        
    # for FID
    def generation_fid(self,path):
        start_time = time.time()
        np.random.seed(595)
        z_fid = np.random.normal(size=[10000, self.n_latent])
        generation_fid = self.generator(z_fid)
        np.save(path+"generation_fid.npy",generation_fid)
        np.save(path+"gen_fid_time.npy",np.array(time.time()-start_time))
        
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.dec, feed_dict={self.X_in: x, self.isTrain: False})
        return x_hat

    # z -> x_hat
    def generator(self, z):
        x_hat = self.sess.run(self.dec, feed_dict={self.sampled: z, self.isTrain: False})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.sampled, feed_dict={self.X_in: x, self.isTrain: False})
        return z    

if __name__ == "__main__":
    start_time = time.time()
    
    root = 'rcnnvae_sp_noise/'
    if not os.path.isdir(root):
        os.mkdir(root)
    
    if len(sys.argv)>1:
        noise_factors = [float(sys.argv[1])]
    else:
#        noise_factors = [round(0.01*i,2) for i in range(1,52,4)]
        noise_factors = [0.2]
        
        
        
    batch_size = 200
    num_epoch = 20 #30*20
    n_latent = 100
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False,reshape=[])
    x_train = mnist.train.images
    X_in = tf.image.resize_images(x_train, [64, 64]).eval()
    
    for noise_factor in noise_factors:
        x_train_noisy = corrupt(X_in, corNum = int(noise_factor*64*64))
        
        tf.reset_default_graph()
        with tf.Session() as sess:
            cnnvae = Deep_CNNVAE(sess = sess, learning_rate = 2e-4, n_latent = n_latent)
            cnnvae.fit(x_train_noisy, path = root, file_name="", num_epoch = num_epoch, batch_size = batch_size)
            