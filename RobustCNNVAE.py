#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 20:05:29 2018

@author: huiminren
"""

import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from BasicAutoencoder import DeepVAE as VAE
from shrink import l1shrink as SHR 

import time
from collections import Counter
import matplotlib.pyplot as plt 

import os

class RVDAE(object):
    """
    @Original author: Chong Zhou
    
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, input_dim,learning_rate = 1e-3, n_z = 5, 
                 lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.errors = []
        self.lambda_ = lambda_
        self.error = error
        self.vae = VAE.VariantionalAutoencoder(sess = sess, input_dim = input_dim, 
                                               learning_rate = learning_rate, n_z = n_z)
        

    def fit(self, X, path = "", num_gen=10, iteration=20, num_epoch = 100, batch_size=64, verbose=False):
        
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        
        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X,'fro')
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
            self.vae.fit(X = self.L, path = path, file_name = "vae_loss"+str(it)+".npy",
                         num_epoch = num_epoch, batch_size = batch_size)
            ## get optmized L
            self.L = self.vae.reconstructor(self.L)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm
            self.errors.append(c1)
            
            if it % 1 == 0 :
                print("generation images:")
                self.vae.plot(FLAG_gen = True, x="", num_gen=num_gen, 
                              path=path, 
                              fig_name="generator_"+str(it)+".png")
#                print("reconstruction images:")
#                self.vae.plot(FLAG_gen = False, x=self.L[:100], num_gen=num_gen, 
#                              path=path, 
#                              fig_name="reconstructor_"+str(it)+".png")
            
            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
        
        return self.L , self.S, np.array(self.errors)
    
    # x --> z
    def transform(self, X):
        L = X - self.S
        return self.vae.transformer(L)
    
    # x -> x_hat
    def getRecon(self, X):
#        L = X - self.S
        return self.vae.reconstructor(self.L)
    
    # z -> x
    def generator(self, z):
        return self.vae.generator(z)
    
    
 
def main(noise_factors,debug = True):
            
    start_time = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train = mnist.train.images
    
    n_z = 5
    
    batch_size = 256
    iteration = 30
    num_epoch = 30
    num_gen = 100
    if debug:
        batch_size = 64
        iteration = 2
        num_epoch = 2
        num_gen = 10
    
    
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "./save_images_l10_rvae/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        
        tf.reset_default_graph()
        sess = tf.Session()
        rvae = RVDAE(sess = sess, input_dim = x_train_noisy.shape[1],learning_rate = 1e-3, n_z = n_z, 
                     lambda_=10, error = 1.0e-7)
        L, S, errors = rvae.fit(X = x_train_noisy, path = path, 
                                num_gen = num_gen,
                                iteration=iteration, num_epoch = num_epoch, 
                                batch_size=batch_size, verbose=True)
        
        x_axis = np.arange(len(errors))
        plt.plot(x_axis, errors, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Plot of trainloss')
        plt.savefig(path+"rvae_loss.png")
        plt.show()
        
        np.save(path+"rvae_errors.npy",errors)
        rvae_recon = rvae.getRecon(x_train_noisy) # get reconstruction
        rvae_transform = rvae.transform(x_train_noisy) # get transformer

        
        np.save(path+"rvae_recon.npy",rvae_recon)
        np.save(path+"rvae_transform.npy",rvae_transform)
        
        sess.close()
        print ("number of zero values in S:", Counter(S.reshape(S.size))[0])
        

    print ('Done_running time:',time.time()-start_time)
    
if __name__ == "__main__":
    noise_factors = np.array([0,0.2,0.4])
    main(noise_factors = noise_factors,debug = False)