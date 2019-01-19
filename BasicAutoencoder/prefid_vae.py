#!/usr/bin/env python3
import sys
import os
import numpy as np
import fid
import tensorflow as tf


def main(nr):
    input_path = 'vae_sp_noise/'
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")
    
    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    
    path = "./fid_precalc_1w/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    data_path = input_path+str(nr)+'/generation_fid.npy'
    images = np.load(data_path)[:10000]
    images = np.stack((((images*255)).reshape(-1,28,28),)*3,axis=-1)
    
    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")
    
    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        output_path = 'fid_stats_noise_'+str(nr)
        np.savez_compressed(path+output_path, mu=mu, sigma=sigma)
    print("finished")

if __name__ == "__main__":
    if len(sys.argv)>1:
        nr = float(sys.argv[1])
    else: 
        nr = 0.2
    main(nr)