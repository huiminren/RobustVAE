#!/usr/bin/env python3
import os
import numpy as np
import fid
import tensorflow as tf


def main(model, noise_factors, lambdas):
    """
    model: RVAE or VAE
    noise_factors: noise factors
    lambdas: lambda
    """
    
    input_path = model
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")
    
    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    
    output_path = "fid_precalc/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for l in lambdas:
        for nr in noise_factors:
            data_path = input_path+'lambda_'+str(l)+'/noise_'+str(nr)+'/generation_fid.npy'
            output_name = 'fid_stats_lambda_'+str(l)+'noise_'+str(nr)
            images = np.load(data_path)
            images = np.transpose(images*255,(0,2,3,1))
            #images = np.stack((((images*255)).reshape(-1,28,28),)*3,axis=-1)
            
            print("create inception graph..", end=" ", flush=True)
            fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
            print("ok")
            
            print("calculte FID stats..", end=" ", flush=True)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
                np.savez_compressed(output_path+output_name, mu=mu, sigma=sigma)
            print("finished")

if __name__ == "__main__":
    models = ['rcnnvae_face/']
    lambdas = [300]
    noise_factors = [0.2,0.3]
    for model in models:
        main(model, noise_factors, lambdas)
