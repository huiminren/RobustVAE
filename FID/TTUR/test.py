#!/usr/bin/env python3

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########
# PATHS
########


input_path = './GAN_sameStructure_result'
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

path = "./fid_precalc_GAN/"

nr = 0.0
data_path = input_path+'/MNIST_GAN_results'+str(nr)+'/saved_result4FID.npy'
images = np.load(data_path)
images = np.reshape(images, [-1, 28, 28])
images = images[0:10000]
images = np.stack((images,)*3, axis=-1)


# print("%d images found and loaded" % len(images))

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
