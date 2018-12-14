import numpy as np
import numpy.linalg as nplin
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from BasicAutoencoder import DeepAE as DAE
from shrink import l1shrink as SHR 
import matplotlib.pyplot as plt 

import time
from collections import Counter
import os

class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-7):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]
        self.AE = DAE.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def fit(self, X, sess, path = "", num_gen = 10, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=50, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]

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
            self.AE.fit(X = self.L, path = path, file_name = "dae_loss"+str(it)+".npy",
                                    sess = sess,
                                    iteration = inner_iteration,
                                    learning_rate = learning_rate,
                                    batch_size = batch_size,
                                    verbose = verbose)
            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S
            self.S = SHR.shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm
            self.errors.append(c1)
            
            if it % 1 == 0 :
                print("generation images:")
                self.AE.plot( num_gen=num_gen, sess=sess,
                              path=path, fig_name="generator_"+str(it)+".png")
            
            if verbose:
                print ("c1: ", c1)
                print ("c2: ", c2)

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S
            
        return self.L , self.S, np.array(self.errors)
    
    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)
    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess = sess)

def main(noise_factors,debug = True):
    start_time = time.time()
    
    batch_size = 256
    iteration = 30
    num_epoch = 30
    num_gen = 100
    if debug:
        batch_size = 64
        iteration = 2
        num_epoch = 2
        num_gen = 10
        
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train = mnist.train.images
    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "./save_images_generator_dae/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)
            
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)


        sess = tf.Session()
        rae = RDAE(sess = sess, lambda_= 2000, layers_sizes=[784,196,49])

        L, S, errors = rae.fit(x_train_noisy, sess = sess, path = path,num_gen = num_gen, learning_rate=0.01, 
                               inner_iteration = iteration, iteration=num_epoch, batch_size = batch_size, 
                               verbose=True)
        
        x_axis = np.arange(len(errors))
        plt.plot(x_axis, errors, 'r-')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Plot of trainloss')
        plt.savefig(path+"dae_loss.png")
        plt.show()
        
        np.save(path+"rdae_errors.npy",errors)
        
        
        recon_rae = rae.getRecon(x_train_noisy, sess = sess)
        trans_rae = rae.transform(x_train_noisy, sess = sess)
        np.save(path+"dae_transform.npy",trans_rae)
        np.save(path+"dae_recon.npy",recon_rae)

        sess.close()
        print ("number of zero values in S:", Counter(S.reshape(S.size))[0])
        print()
    print ('Done_running time:',time.time()-start_time)

if __name__ == "__main__":
#    noise_factors = np.arange(0.1)
    noise_factors = np.array([0.2,0.4])
    main(noise_factors = noise_factors, debug = False)
