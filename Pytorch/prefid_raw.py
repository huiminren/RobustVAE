#!/usr/bin/env python3
import os
import glob
import numpy as np
import fid
import tensorflow as tf
from utils import *


def main():
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")
    
    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    
    data_files = glob.glob(os.path.join("./img_align_celeba", "*.jpg"))
    data_files = sorted(data_files)[:10000]
    data_files = np.array(data_files)
    images = np.array([get_image(data_file, 148) for data_file in data_files]).astype(np.float32)
    images = images*255
    
    output_name = 'fid_stats_face'

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_name, mu=mu, sigma=sigma)
    print("finished")

if __name__ == "__main__":
    main()
