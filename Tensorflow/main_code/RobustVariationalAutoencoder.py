#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:40:28 2019

@author: Huimin Ren
"""

import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from BasicAutoencoder import DeepVAE as VAE
from shrink import l1shrink as SHR 

import time
import sys

import os

def corrupt(X_in,corNum=10):
    X = X_in.copy()
    N,p = X.shape[0],X.shape[1]
    for i in range(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j] > 0.5:
                X[i,j] = 0
            else:
                X[i,j] = 1
    return X

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
        input_dim: input dimension
        learning_rate: learning rate for optimizing VAE
        n_z: dimension of latent layer(coder)
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.errors = []
        self.lambda_ = lambda_
        self.error = error
        self.vae = VAE.VariantionalAutoencoder(sess = sess, input_dim = input_dim, 
                                    learning_rate = learning_rate, n_z = n_z)
        

    def fit(self, X, path = "", num_gen=10, iteration=20, num_epoch = 100, 
            batch_size=64, verbose=False):
        
        """
        X: input data
        path: path of saving loss and generation
        num_gene: number of generated images
        iteration: number of outer iteration
        num_epoch: number of epoch for each VAE (inner iteration)
        batch_size: batch size of VAE
        """
        
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
            self.vae.fit(X = self.L, path = path, file_name = "vae_loss"+str(it),
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
            
            # generate sample images for visual check and FID computation
            if it == iteration-1:
                print("generation images:")
                self.vae.gen_plot(FLAG_gen = True, x="", num_gen=num_gen, 
                              path=path, fig_name="generator_"+str(it)+".png")
                print("generate fid images")
                self.vae.generation_fid(path=path)
            
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
    
    
 
def main(data_source, noise_method, noise_factors,lambdas, debug = True):
    """
    data_source: data set of training. Either 'MNIST' or 'FASHION'
    noise_method: method of adding noise. Either 'sp' (represents salt-and-pepper) 
                  or 'gs' (represents Gaussian)
    noise_factors: noise factors
    lambdas: multiple values of lambda
    debug: True or False
    """
    
    if data_source == 'MNIST':
        data_dir = 'input_data/MNIST_data/'
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
    if data_source == 'FASHION':
        data_dir = 'input_data/Fashion_data/'
        mnist = input_data.read_data_sets(data_dir, 
                source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    
    x_train = mnist.train.images
    
    n_z = 49
    batch_size = 200
    iteration = 30
    num_epoch = 20
    num_gen = 100
    if debug:
        x_train = mnist.train.images[:1000]
        batch_size = 64
        iteration = 2
        num_epoch = 2
        num_gen = 10
    
    output = "output/"
    if not os.path.exists(output):
        os.mkdir(output)
            
    for lambda_ in lambdas:
        print("lambda:",lambda_)
        path = output+"RVAE_"+data_source+"_"+noise_method+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+"lambda_"+str(lambda_)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
        for noise_factor in noise_factors:
            print("noise factor: ",noise_factor)
            path = output+"RVAE_"+data_source+"_"+noise_method+"/"+"lambda_"+str(lambda_)+"/"
            path = path+"noise_"+str(noise_factor)+"/"
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
            sess = tf.Session()
            rvae = RVDAE(sess = sess, input_dim = x_train_noisy.shape[1],
                learning_rate = 1e-3, n_z = n_z, lambda_=lambda_, error = 1.0e-7)
            L, S, errors = rvae.fit(X = x_train_noisy, path = path, 
                                    num_gen = num_gen,
                                    iteration=iteration, num_epoch = num_epoch, 
                                    batch_size=batch_size, verbose=True)
            
            sess.close()
            np.save(path+'running_time.npy',np.array(time.time()-start_time))
    
if __name__ == "__main__":
    if len(sys.argv)>4:
        data_source = sys.argv[1]
        noise_method = sys.argv[2]
        noise_factors = [float(sys.argv[3])]
        lambdas = [int(sys.argv[4])]
        main(data_source, noise_method, noise_factors,lambdas, debug = True)
        
    else:
        lambdas = [1,5,10,15,20,25,50,70,100,250]
        if noise_method == 'sp':
            noise_factors = [round(i*0.01,2) for i in range(1,52,2)]
        if noise_method == 'gs':
            noise_factors = [round(i*0.1,1) for i in range(1,10)]
        data_sources = ['MNIST','FASHION']
        noise_methods = ['sp','gs']
        for data_source in data_sources:
            for noise_method in noise_methods:
                main(data_source, noise_method, noise_factors,lambdas, debug = True)