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

import os
import sys

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

class RCNNVDAE(object):
    """
    @Original author: Chong Zhou
    
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, learning_rate = 1e-3, n_latent = 8, 
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
        self.cnnvae = CNNVAE.Deep_CNNVAE(sess = sess, learning_rate = learning_rate, n_latent = n_latent)
        
    def fit(self, X, path = "", num_gen=10, iteration=20, num_epoch = 100, batch_size=64, verbose=False):
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
                         num_epoch = num_epoch, batch_size = batch_size)
            ## get optmized L
            self.L = self.cnnvae.reconstructor(self.L)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm((X - self.L - self.S).reshape(-1,X.shape[-1]*X.shape[-1]), 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm
            self.errors.append(c1)
            
            if it % 1 == 0 :
                print("generation images:")
                self.cnnvae.gen_plot(FLAG_gen = True, x="", num_gen=num_gen, 
                              path=path, fig_name="generator_"+str(it)+".png")
                
            if it == iteration-1:
                print("generate fid images")
                self.cnnvae.generation_fid(path=path)
                
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
    def transform(self, X):
        L = X - self.S
        return self.cnnvae.transformer(L)
    
    # x -> x_hat
    def getRecon(self, X):
#        L = X - self.S
        return self.cnnvae.reconstructor(self.L)
    
    # z -> x
    def generator(self, z):
        return self.cnnvae.generator(z)
    
    
 
def main(noise_factors, lambdas, debug = True):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False,reshape=[])
    X_in = mnist.train.images
    x_train = tf.image.resize_images(X_in, [64, 64]).eval()

    
    batch_size = 200
    iteration = 30
    num_epoch = 20
    num_gen = 100
    if debug:
        batch_size = 64
        iteration = 2
        num_epoch = 2
        num_gen = 10
    
    
    for lambda_ in lambdas:
        print("lambda:",lambda_)
        path = "./rcnnvae_sp_noise/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+"lambda_"+str(lambda_)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        for noise_factor in noise_factors:
            print("noise factor: ",noise_factor)
            path = "./rcnnvae_sp_noise/"+"lambda_"+str(lambda_)+"/"
            path = path+"noise_"+str(noise_factor)+"/"
            if not os.path.exists(path):
                os.mkdir(path)
                
            start_time = time.time()
            np.random.seed(595)
#            x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#            x_train_noisy = np.clip(x_train_noisy, 0., 1.)
            x_train_noisy = corrupt(x_train, corNum = int(noise_factor*64*64))
            
            tf.reset_default_graph()
            sess = tf.Session()
            rcnnvae = RCNNVDAE(sess = sess,learning_rate = 2e-4, n_latent = 100, 
                         lambda_=100, error = 1.0e-7)
            
            L, S, errors = rcnnvae.fit(X = x_train_noisy, path = path, 
                                    num_gen = num_gen,
                                    iteration=iteration, num_epoch = num_epoch, 
                                    batch_size=batch_size, verbose=True)
            
            sess.close()
            print ("number of zero values in S:", Counter(S.reshape(S.size))[0])
            np.save(path+'running_time.npy',np.array(time.time()-start_time))
            
    
if __name__ == "__main__":
    if len(sys.argv)>2:
        lambdas = [int(sys.argv[1])]
        noise_factors = [float(sys.argv[2])]
    else:
#        lambdas = [1,5,10,15,20,25,50,70,100,250]
        lambdas = [50, 70, 100]
#        noise_factors = [round(i*0.01,2) for i in range(1,52,4)]
        noise_factors = [0.05, 0.13, 0.23]
#        lambdas = [50]
#        noise_factors = [0.21]
    #noise_factors = [0.1]
    #lambdas = [0.1,1,5,10,15,20,25,50,70,100,250]
    main(noise_factors,lambdas,debug = False)