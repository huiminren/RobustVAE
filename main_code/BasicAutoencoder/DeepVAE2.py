#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:47:48 2018

@author: Huimin Ren
Fork from: https://github.com/shaohua0116/VAE-Tensorflow/blob/master/demo.ipynb
           changed activation function and the dimension of latent layer 
"""


import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
import sys

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))

def corrupt(X,corNum=10):
    """Method of adding salt-and-pepper noise"""
    N,p = X.shape[0],X.shape[1]
    for i in range(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j] > 0.5:
                X[i,j] = 0
            else:
                X[i,j] = 1
    return X
        
class VariantionalAutoencoder(object):

    def __init__(self, sess, input_dim = 784, learning_rate=1e-3, n_z=49):
        """
        sess: tf.Session()
        input_dim: input dimension
        learning_rate: learning rate for optimizer
        n_z: dimension for latent layer(coder)
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_z = n_z

        self.build()

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.sigmoid)
        f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.sigmoid)
        f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.sigmoid)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.exp(self.z_log_sigma_sq) * eps # remove sqrt

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 256, scope='dec_fc1', activation_fn=tf.nn.sigmoid)
        g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.sigmoid)
        g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.sigmoid)
        self.x_hat = fc(g3, self.input_dim, scope='dec_fc4', activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat), 
            axis=1
        )
        
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        """
        x: data to be fit
        """
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )
        return loss, recon_loss, latent_loss
    
    
    def fit(self, X, path = "", file_name="", num_epoch = 100, batch_size = 64):
        """
        X: data to be fit
        num_epoch: number of epoch
        batch_size: batch size
        """
        ls_loss = []
        re_loss = []
        la_loss = []
        sample_size = X.shape[0]
        for epoch in range(num_epoch):
            ls_tmp = []
            re_tmp = []
            la_tmp = []
            for one_batch in batches(sample_size, batch_size):
                loss, recon_loss, latent_loss = self.run_single_step(X[one_batch])
                ls_tmp.append(loss)
                re_tmp.append(re_loss)
                la_tmp.append(latent_loss)
                
            ls_loss.append(np.mean(ls_tmp))
            re_loss.append(np.mean(re_tmp))
            la_loss.append(np.mean(la_tmp))
            if epoch % 1 == 0:
                print('[epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                    epoch, loss, recon_loss, latent_loss))
                
            if epoch == num_epoch-1:
                print("generate images:")
                self.gen_plot(FLAG_gen = True, x = "", num_gen=100, path=path,fig_name='generator'+str(epoch)+'.png')
                print("gid geneartion:")
                self.generation_fid(path = path)
                
        np.save(path+file_name+"_all.npy",np.array(ls_loss))
        np.save(path+file_name+"_recons.npy",np.array(re_loss))
        np.save(path+file_name+"_latent.npy",np.array(la_loss))
        
        
    def gen_plot(self,FLAG_gen = True, x="", num_gen=100, path="", fig_name=""):
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
            z = np.random.normal(size=[num_gen, self.n_z])
            rvae_generated = self.generator(z) # get generation
        else:
            rvae_generated = x
        
        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = rvae_generated[i*n+j, :].reshape(28, 28)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
#        plt.show()
        
    # for FID
    def generation_fid(self,path):
        start_time = time.time()
        np.random.seed(595)
        z_fid = np.random.normal(size=[10000, self.n_z])
        generation_fid = self.generator(z_fid)
        np.save(path+"generation_fid.npy",generation_fid)
        np.save(path+"gen_fid_time.npy",np.array(time.time()-start_time))
        
        
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    
def main(data_source, noise_method, noise_factors, debug = True):
    """
    data_source: data set of training. Either 'MNIST' or 'FASHION'
    noise_method: method of adding noise. Either 'sp' (represents salt-and-pepper) 
                  or 'gs' (represents Gaussian)
    noise_factors: noise factors
    debug: True or False
    """
    if data_source == 'MNIST':
        data_dir = '../input_data/MNIST_data/'
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
    if data_source == 'FASHION':
        data_dir = '../input_data/Fashion_data/'
        mnist = input_data.read_data_sets(data_dir, 
                source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    
    x_train = mnist.train.images
    
    n_z = 49
    batch_size = 200
    num_epoch = 30*2020
    if debug:
        x_train = mnist.train.images[:1000]
        batch_size = 64
        num_epoch = 2
        
    output = "../output/"
    if not os.path.exists(output):
        os.mkdir(output)
        
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = output+"/VAE_"+data_source+"_"+noise_method+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        
        np.random.seed(595)
        if noise_method == 'sp':
            x_train_noisy = corrupt(x_train, corNum = int(noise_factor*784))
        if noise_method == 'gs':
            x_train_noisy = x_train + noise_factor * np.random.normal(
                                loc=0.0, scale=1.0, size=x_train.shape) 
            x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        start_time = time.time() # record running time for training
        tf.reset_default_graph()
        with tf.Session() as sess:
            
            vae = VariantionalAutoencoder(sess = sess, input_dim = x_train_noisy.shape[1], n_z=n_z)
            vae.fit(x_train_noisy, path = path,file_name="loss", 
                    num_epoch = num_epoch, batch_size = batch_size)
        np.save(path+"running_time.np",np.array(time.time()-start_time))
            
if __name__ == "__main__":
    if len(sys.argv)>3:
        data_source = sys.argv[1]
        noise_method = sys.argv[2]
        noise_factors = [float(sys.argv[3])]
        main(data_source, noise_method, noise_factors,debug = True)
    else:
        if noise_method == 'sp':
            noise_factors = [round(i*0.01,2) for i in range(1,52,2)]
        if noise_method == 'gs':
            noise_factors = [round(i*0.1,1) for i in range(1,10)]
        data_sources = ['MNIST','FASHION']
        noise_methods = ['sp','gs']
        for data_source in data_sources:
            for noise_method in noise_methods:
                main(data_source, noise_method, noise_factors,debug = True)
    
    
    