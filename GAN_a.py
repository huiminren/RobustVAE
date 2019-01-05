import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import time
import os

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i, min(l, i + n))


class DCGAN(object):
    def __init__(self, sess, batch_size, input_height, input_width, noise_dim=100, learning_rate=1e-3):
        """
        sess: tf.Session()
        noise_dim: noise dimension (like dimension in latent layer)
        dec_in_channels: dimension in channel
        learning_rate: learning rate for loss
        """

        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.build()

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        # Build Networks
        # Network Inputs
        self.noise_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        # Build Generator Network
        self.gen_sample = self.generator(self.noise_input)

        # Build 2 Discriminator Networks (one from noise input, one from generated samples)
        disc_real = self.discriminator(self.real_image_input)
        disc_fake = self.discriminator(self.gen_sample, reuse=True)
        disc_concat = tf.concat([disc_real, disc_fake], axis=0)

        # Build the stacked generator/discriminator
        stacked_gan = self.discriminator(self.gen_sample, reuse=True)

        # Build Targets (real or fake images)
        self.disc_target = tf.placeholder(tf.int32, shape=[None])
        self.gen_target = tf.placeholder(tf.int32, shape=[None])

        # Build Loss
        self.disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_concat, labels=self.disc_target))
        self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=stacked_gan, labels=self.gen_target))

        # Build Optimizers
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Training Variables for each optimizer
        # By default in TensorFlow, all variables are updated by each optimizer, so we
        # need to precise for each one of them the specific variables to update.
        # Generator Network Variables
        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        # Discriminator Network Variables
        self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        # Create training operations
        self.train_gen = self.optimizer_gen.minimize(self.gen_loss, var_list=self.gen_vars)
        self.train_disc = self.optimizer_disc.minimize(self.disc_loss, var_list=self.disc_vars)

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))

    # G(z)
    def generator(x, reuse=False):
        # initializers
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
        b0 = tf.get_variable('G_b0', [256], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

        # 2nd hidden layer
        w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init)
        b1 = tf.get_variable('G_b1', [512], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

        # 3rd hidden layer
        w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init)
        b2 = tf.get_variable('G_b2', [1024], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # output hidden layer
        w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 784], initializer=w_init)
        b3 = tf.get_variable('G_b3', [784], initializer=b_init)
        o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        return o

    # D(x)
    def discriminator(x, reuse=False):

        # initializers
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
        b0 = tf.get_variable('D_b0', [1024], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

        # 2nd hidden layer
        w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init)
        b1 = tf.get_variable('D_b1', [512], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

        # 3rd hidden layer
        w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
        b2 = tf.get_variable('D_b2', [256], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # output layer
        w3 = tf.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
        b3 = tf.get_variable('D_b3', [1], initializer=b_init)
        o = tf.sigmoid(tf.matmul(h2, w3) + b3)

        return o

    '''
    # Generator Network
    def generator(self, X_in, reuse = False):
        """
        Input: X_in: Noise; reuse: Boolean
        Output: Image
        """
        with tf.variable_scope('Generator', reuse=reuse):
            # TensorFlow Layers automatically create variables and calculate their
            # shape, based on the input.
            x = tf.layers.dense(X_in, units=256)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, units=384)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, units=512)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, units=784)
            x = tf.nn.sigmoid(x)
            # Reshape to a 4-D array of images: (batch, height, width, channels)
            # New shape: (batch, 6, 6, 128)
            # x = tf.reshape(x, shape=[-1, 28, 28, 1])
            # # Deconvolution, image shape: (batch, 14, 14, 64)
            # x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
            # # Deconvolution, image shape: (batch, 28, 28, 1)
            # x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
            # # Apply sigmoid to clip values between 0 and 1
            # x = tf.nn.sigmoid(x)
        return x
    # Discriminator Network
        # Discriminator Network

    def discriminator(self, X_in, reuse=False):
        """
        Input: X_in: Image; reuse: boolean
        Output: Prediction Real/Fake Image
        """
        with tf.variable_scope('Discriminator', reuse=reuse):
            # Typical convolutional neural network to classify images.
            # x = tf.layers.conv2d(X_in, 64, 5)
            # x = tf.nn.sigmoid(x)
            x = tf.contrib.layers.flatten(X_in)
            x = tf.layers.dense(x, 512)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, 384)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, 256)
            x = tf.nn.sigmoid(x)
            x = tf.layers.dense(x, 5)
            x = tf.nn.sigmoid(x)
            # Output 2 classes: Real and Fake images
            x = tf.layers.dense(x, 2)
        return x
    '''

    def fit(self, X_in, path, num_gen=100, num_epoch=100, batch_size=64):
        """
        X_in: image without label
        """
        sample_size = X_in.shape[0]
        X_in = np.reshape(X_in, [-1, 28, 28, 1])
        gl_loss = []
        dl_loss = []

        for epoch in range(num_epoch):
            tmp_gl = []
            tmp_dl = []
            for one_batch in batches(sample_size, batch_size):
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, self.noise_dim])
                # Prepare Targets (Real image: 1, Fake image: 0)
                # The first half of data fed to the generator are real images,
                # the other half are fake images (coming from the generator).
                batch_disc_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
                # Generator tries to fool the discriminator, thus targets are 1.
                batch_gen_y = np.ones([batch_size])
                # Training
                feed_dict = {self.real_image_input: X_in[one_batch], self.noise_input: z,
                             self.disc_target: batch_disc_y, self.gen_target: batch_gen_y}
                _, _, gl, dl = self.sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                             feed_dict=feed_dict)
                tmp_gl.append(gl)
                tmp_dl.append(dl)
            gl_loss.append(np.mean(tmp_gl))
            dl_loss.append(np.mean(tmp_dl))
            if epoch % 1 == 0:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (epoch, np.mean(tmp_gl), np.mean(tmp_dl)))
                self.plot(path=path, fig_name="generator_" + str(epoch) + ".png", num_gen=num_gen)

        np.save(path + "gl_loss.npy", np.array(gl_loss))
        np.save(path + "dl_loss.npy", np.array(dl_loss))

    def plot(self, path, fig_name, num_gen=100):

        np.random.seed(595)
        h = w = 28
        z = np.random.uniform(-1., 1., size=[num_gen, self.noise_dim])
        g = self.get_generation(z)

        # plot of generation
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h * n, w * n))
        for i in range(n):
            for j in range(n):
                I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = g[i * n + j, :].reshape(28, 28)

        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path + fig_name)

    def get_generation(self, z):
        return self.sess.run(self.gen_sample, feed_dict={self.noise_input: z})


def main(noise_factors, debug=True):
    start_time = time.time()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    x_train = mnist.train.images

    # input and output parameter
    input_height = 28
    input_width = 28
    nois_dim = 5

    batch_size = 200  # X_in.shape[0] % batch_size == 0
    num_epoch = 30
    num_gen = 100
    if debug:
        x_train = x_train[:100]
        batch_size = 10
        num_epoch = 30
        num_gen = 10

    for noise_factor in noise_factors:
        print("noise factor: ", noise_factor)
        path = "save_images_GAN_a/"
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + str(noise_factor) + "/"
        if not os.path.exists(path):
            os.mkdir(path)

        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        tf.reset_default_graph()
        sess = tf.Session()
        dcgan = DCGAN(sess,  batch_size=batch_size, input_height=input_height, input_width=input_width,
                      noise_dim=nois_dim,
                      learning_rate=1e-3)
        dcgan.fit(x_train_noisy, path=path, num_gen=num_gen, num_epoch=num_epoch, batch_size=batch_size)
        sess.close()

    print("running time: ", time.time() - start_time)


if __name__ == "__main__":
    noise_factors = np.array([0.2, 0.4])
    main(noise_factors=noise_factors, debug=True)