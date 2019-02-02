#!/usr/bin/env python3
import sys
import os
import numpy as np
import fid
import tensorflow as tf


def main(model, data_source, noise_method, noise_factors, lambdas):
    """
    model: RVAE or VAE
    data_source: data set of training. Either 'MNIST' or 'FASHION'
    noise_method: method of adding noise. Either 'sp' (represents salt-and-pepper) 
                  or 'gs' (represents Gaussian)
    noise_factors: noise factors
    lambdas: lambda
    """
    
    input_path = "../output/"+model+"_"+data_source+"_"+noise_method+"/"
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")
    
    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    
    output_path = "fid_precalc/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = output_path+model+"_"+data_source+"_"+noise_method+"/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for l in lambdas:
        for nr in noise_factors:
            if model == 'RVAE':
                data_path = input_path+'lambda_'+str(l)+'/noise_'+str(nr)+'/generation_fid.npy'
                output_name = 'fid_stats_lambda_'+str(l)+'noise_'+str(nr)
            else:
                data_path = input_path+str(nr)+'/generation_fid.npy'
                output_name = 'fid_stats_noise_'+str(nr)
            images = np.load(data_path)[:10000]
            images = np.stack((((images*255)).reshape(-1,28,28),)*3,axis=-1)
            
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
    if len(sys.argv) == 6:
        model = sys.argv[1]
        data_source = sys.argv[2]
        noise_method = sys.argv[3]
        noise_factors = int(sys.argv[4])
        lambdas = float(sys.argv[5])
        main(model, data_source, noise_method, noise_factors, lambdas)
    else:
        models = ['RVAE','VAE']
        data_sources = ['MNIST','FASHION']
        noise_methods = ['sp','gs']
        for model in models:
            if model == 'RVAE':
                lambdas = [1,5,10,15,20,25,50,70,100,250]
            else:
                lambdas = [1]
            for data_source in data_sources:
                for noise_method in noise_methods:
                    if noise_method == 'sp':
                        noise_factors = [round(i*0.01,2) for i in range(1,52,2)]
                    else:
                        noise_factors = [round(i*0.1,1) for i in range(1,10)]
                    
                    main(model, data_source, noise_method, noise_factors, lambdas)