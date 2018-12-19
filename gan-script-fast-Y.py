
"""
##################
Author: Angela Y
This code is modified from https://github.com/jonbruner/generative-adversarial-networks to accommodate our research study
##################

This is a straightforward Python implementation of a generative adversarial network.
The code is derived from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

The tutorial's code trades efficiency for clarity in explaining how GANs function;
this script refactors a few things to improve performance, especially on GPU machines.
In particular, it uses a TensorFlow operation to generate random z values and pass them
to the generator; this way, more computations are contained entirely within the
TensorFlow graph.

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import datetime
import numpy as np
# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.contrib.slim import fully_connected as fc


# Define the discriminator network
def discriminator(x, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        f1 = fc(x, 512, scope='enc_fc1', activation_fn=tf.nn.sigmoid)
        f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.sigmoid)
        f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.sigmoid)
        out = fc(f3, 1, scope="enc_fc4", activation_fn=tf.nn.sigmoid)

        # contains unscaled values
        return out

# Define the generator network
def generator(batch_size, z_dim, input_dim):
    z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')

    g1 = fc(z, 256, scope='dec_fc1', activation_fn=tf.nn.sigmoid)
    g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.sigmoid)
    g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.sigmoid)
    out = fc(g3, input_dim, scope='dec_fc4', activation_fn=None)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return out

def main(noise_factors, debug=True):
    np.random.seed(595)

    start_time = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    learning_rate = 1e-3
    input_dim = 784
    z_dimensions = 49

    batch_size = 256
    num_epoch = 30
    num_gen = 100
    
    if debug:
        batch_size = 64
        num_epoch = 2
        num_gen = 10



    for noise_factor in noise_factors:
        print("noise factor: ",noise_factor)
        path = "./save_images_l100_gan/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path+str(noise_factor)+"/"
        if not os.path.exists(path):
            os.mkdir(path)

        x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
	    # x_placeholder is for feeding input images to the discriminator

        Gz = generator(batch_size, z_dimensions, input_dim)
	    # Gz holds the generated images

        Dx = discriminator(x_placeholder)
	    # Dx will hold discriminator prediction probabilities
	    # for the real MNIST images

        Dg = discriminator(Gz, reuse_variables=True)
	    # Dg will hold discriminator prediction probabilities for generated images

	    # Define losses
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

	    # Define variable lists
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

	    # Train the discriminator
        d_trainer_fake = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_fake, var_list=d_vars)
        d_trainer_real = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_real, var_list=d_vars)

	    # Train the generator
        g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

	    # From this point forward, reuse variables
        tf.get_variable_scope().reuse_variables()

        sess = tf.Session()

	    # Send summary statistics to TensorBoard
        tf.summary.scalar('Generator_loss', g_loss)
        tf.summary.scalar('Discriminator_loss_real', d_loss_real)
        tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

        images_for_tensorboard = generator(batch_size, z_dimensions)
        tf.summary.image('Generated_images', images_for_tensorboard, 5)
        merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(tf.global_variables_initializer())


	    # Train generator and discriminator together
        for i in range(num_epoch):
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            x_train_noisy = real_image_batch + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=real_image_batch.shape)
            x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    	    # Train discriminator on both real and fake images
            _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: x_train_noisy})

    	    # Train generator
            _ = sess.run(g_trainer)

        if i % 10 == 0:
        	# Update TensorBoard with summary statistics
            summary = sess.run(merged, {x_placeholder: x_train_noisy})
            writer.add_summary(summary, i)


if __name__ == "__main__":
    noise_factors = np.array([0.2,0.4])
    main(noise_factors = noise_factors,debug = False)
