#!/usr/bin/env python3

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########
# PATHS
########
lambda_parameter = [0.1,1,5,10,15,20,25,50,70,100]
#noise_ratio = [round(0.1*x, 1) for x in range(11)]
noise_factors = [round(0.1*i,1) for i in range(11)]
noise_factors.extend([round(0.01*i,2) for i in range(51,60)])
noise_factors.extend([round(0.01*i,2) for i in range(61,70)])
noise_factors.sort()


input_path = 'rvae'
#data_path = '../../MNIST_data/' # set path to training set images
#output_path = 'fid_stats_MNIST.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
'''
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
'''
'''
mnist = input_data.read_data_sets(data_path, one_hot=True)
images = mnist.train.images
'''

path = "./fid_precalc/"
if not os.path.exists(path):
    os.mkdir(path)
    
for l in lambda_parameter:
    for nr in noise_factors:
        data_path = input_path+'/lambda_'+str(l)+'/noise_'+str(nr)+'/generation_fid.npy'
        tmp = np.load(data_path)
        images = np.stack((((tmp*255)).reshape(-1,28,28),)*3,axis=-1)
#        images = np.reshape(images, [-1, 28, 28])
#        images = images[0:10000]
#        images = np.stack((images,)*3, axis=-1)
        
        
# print("%d images found and loaded" % len(images))

        print("create inception graph..", end=" ", flush=True)
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        print("ok")

        print("calculte FID stats..", end=" ", flush=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
            output_path = 'fid_stats_lambda_'+str(l)+'noise_'+str(nr)
            np.savez_compressed(path+output_path, mu=mu, sigma=sigma)
        print("finished")


########################################################################################################
# save real image
########################################################################################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
images = mnist.train.images[:10000]
images = np.stack((((images*255)).reshape(-1,28,28),)*3,axis=-1)


inception_path = fid.check_or_download_inception(None) # download inception network
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed("fid_stats_mnist", mu=mu, sigma=sigma)
    
    