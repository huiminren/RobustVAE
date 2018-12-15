#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:05:29 2018

@author: huiminren
"""

import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from BasicAutoencoder import DeepVAE_cnn as CNNVAE
from shrink import l1shrink as SHR 

import time
from collections import Counter
import matplotlib.pyplot as plt 

import os

class RCNNVDAE(object):
    """
    @Original author: Chong Zhou
    
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, learning_rate = 1e-3, n_latent = 8, input_dim = 28,dec_in_channels = 1,
                 lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        learning_rate: CNNVAE learning rate
        n_latent: number of neurons in the latent layer
        input_dim: input dimension as a matrix
        dec_in_channels: number of channels
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.errors = []
        self.lambda_ = lambda_
        self.error = error
        self.cnnvae = CNNVAE.Deep_CNNVAE(sess = sess, learning_rate = learning_rate, n_latent = n_latent,
                   input_dim = input_dim, dec_in_channels = dec_in_channels)
        
    def fit(self, X, path = "", num_gen=10, iteration=20, num_epoch = 100, batch_size=64, 
            keep_prob = 1.0, verbose=False):
        # initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        
        # since the input dimension of nplin must be 1D or 2D, change the shape only for nplin.
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.norm.html
        X_norm=X.reshape(-1,X.shape[-1]*X.shape[-1])
        
        mu = (X_norm.size) / (4.0 * nplin.norm(X_norm,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X_norm,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("mu: ", mu)
            print ("XFnorm: ", XFnorm)
        
        
        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder
            self.cnnvae.fit(X_in = self.L, path = path, file_name = "vae_loss"+str(it)+".npy",
                         num_epoch = num_epoch, batch_size = batch_size,keep_prob = keep_prob)
            ## get optmized L
            self.L = self.cnnvae.reconstructor(self.L,keep_prob)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm((X - self.L - self.S).reshape(-1,X.shape[-1]*X.shape[-1]), 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm
            self.errors.append(c1)
            
            print("generation images:")
            self.cnnvae.plot(FLAG_gen = True, x="", num_gen=num_gen, path=path, 
                          fig_name="generator_"+str(it)+".png")
            
            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
        
        return self.L , self.S, np.array(self.errors)
    
    # x -> z
    def transform(self, X, keep_prob):
        L = X - self.S
        return self.cnnvae.transformer(L,keep_prob)
    
    # x -> x_hat
    def getRecon(self, X, keep_prob):
#        L = X - self.S
        return self.cnnvae.reconstructor(self.L,keep_prob)
    
    # z -> x
    def generator(self, z, keep_prob):
        return self.cnnvae.generator(z,keep_prob)
    
    
 
def main(noise_factors,debug = True):
            
    start_time = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train = mnist.train.images
    input_dim = 28
    x_train = x_train.reshape(-1,input_dim,input_dim)
    
    batch_size = 256
    iteration = 30
    num_epoch = 30
    num_gen = 100
    if debug:
        batch_size = 64
        iteration = 1
        num_epoch = 1
        num_gen = 10
    
    
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "./save_images_cnn/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        tf.reset_default_graph()
        sess = tf.Session()
        rcnnvae = RCNNVDAE(sess = sess,learning_rate = 1e-3, n_latent = 8, input_dim = input_dim, dec_in_channels = 1, 
                     lambda_=100, error = 1.0e-7)
        
        L, S, errors = rcnnvae.fit(X = x_train_noisy, path = path, 
                                num_gen = num_gen,
                                iteration=iteration, num_epoch = num_epoch, 
                                batch_size=batch_size, keep_prob = 1.0, verbose=True)
        
        
        x_axis = np.arange(len(errors))
        plt.plot(x_axis, errors, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Plot of trainloss')
        plt.savefig(path+"rvae_loss.png")
        plt.show()
        
        np.save(path+"rvae_errors.npy",errors)
        rvae_recon = rcnnvae.getRecon(x_train_noisy,keep_prob = 1) # get reconstruction
        rvae_transform = rcnnvae.transform(x_train_noisy,keep_prob = 1) # get transformer

        
        np.save(path+"rvae_recon.npy",rvae_recon)
        np.save(path+"rvae_transform.npy",rvae_transform)
        
        sess.close()
        print ("number of zero values in S:", Counter(S.reshape(S.size))[0])
        

    print ('Done_running time:',time.time()-start_time)
    
if __name__ == "__main__":
    noise_factors = np.array([0.2,0.4])
    main(noise_factors = noise_factors,debug = True)