# Reference: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN
import sys, os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
# G(z)
def generator(x):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
    b0 = tf.get_variable('G_b0', [256], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)


    # 2nd hidden layer
    w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 384], initializer=w_init)
    b1 = tf.get_variable('G_b1', [384], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

    # 3rd hidden layer
    w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 512], initializer=w_init)
    b2 = tf.get_variable('G_b2', [512], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output hidden layer
    w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 784], initializer=w_init)
    b3 = tf.get_variable('G_b3', [784], initializer=b_init)
    o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

    return o


# D(x)
def discriminator(x, drop_out):


    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('D_w0', [x.get_shape()[1], 512], initializer=w_init)
    b0 = tf.get_variable('D_b0', [512], initializer=b_init)
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
    h0 = tf.nn.dropout(h0, drop_out)

    # 2nd hidden layer
    w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 384], initializer=w_init)
    b1 = tf.get_variable('D_b1', [384], initializer=b_init)
    h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
    h1 = tf.nn.dropout(h1, drop_out)


    # 3rd hidden layer
    w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init)
    b2 = tf.get_variable('D_b2', [256], initializer=b_init)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2 = tf.nn.dropout(h2, drop_out)

    # output layer
    w3 = tf.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init)
    b3 = tf.get_variable('D_b3', [1], initializer=b_init)
    o = tf.sigmoid(tf.matmul(h2, w3) + b3)

    return o

digit_n = 10
n_sample_from = 49
h = w = 28
np.random.seed(595)
fixed_z_ = np.random.normal(0, 1, (digit_n*digit_n, n_sample_from))
def show_result(sess, G_z, z, drop_out, num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    np.random.seed(595)

    z_ = np.random.normal(0, 1, (digit_n*digit_n, n_sample_from))

    if isFix:
        test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
    else:
        test_images = sess.run(G_z, {z: z_, drop_out: 0.0})

    # size_figure_grid = digit_n

    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(digit_n, digit_n))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)


    # for k in range(digit_n*digit_n):
        # i = k // digit_n
        # j = k % digit_n
        # ax[i, j].cla()
        # ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    # plot of generation

        # plot of generation
    I_generated = np.empty((h * digit_n, w * digit_n))
    for i in range(digit_n):
        for j in range(digit_n):
            k = i*10 + j
            I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = test_images[k].reshape(28, 28)

    plt.figure(figsize=(8, 8))
    plt.imshow(I_generated, cmap='gray')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_noise(test_images, path, noise_factor, show=False):

    # size_figure_grid = digit_n
    # fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(digit_n, digit_n))
    # for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for k in range(digit_n*digit_n):
    #     i = k // digit_n
    #     j = k % digit_n
    #     ax[i, j].cla()
    #     ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
    #
    # label = 'Noise Factor {}'.format(noise_factor)
    # fig.text(0.5, 0.04, label, ha='center')
    # plt.savefig(path)

    I_generated = np.empty((h * digit_n, w * digit_n))
    for i in range(digit_n):
        for j in range(digit_n):
            k = i*10 + j
            I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = test_images[k].reshape(28, 28)

    plt.figure(figsize=(8, 8))
    plt.imshow(I_generated, cmap='gray')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()



def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def main(noise_factor):

    # training parameters
    batch_size = 200
    lr = 0.0002
    train_epoch = 600
    n_4_FID = 10000
    noise_factor_S = 'SP' + str(noise_factor)
    np.random.seed(595)

    data_source = 'notMNIST'
    # load data
    if data_source == 'fashion':
        data_dir = 'input_data/fashion_MNIST/'
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
        train_set = mnist.train.images

    elif data_source == 'MNIST':
        data_dir = 'input_data/MNIST_data/'
        mnist = input_data.read_data_sets(data_dir, one_hot=True)
        train_set = mnist.train.images

    elif data_source == 'notMNIST':
        data_dir = './notMNIST2.npy'
        train_set = np.load(data_dir)

    train_set = corrupt(train_set, corNum=int(noise_factor * 784))


    # plot image after adding noise
    if not os.path.isdir(data_source+'_GAN_results'+noise_factor_S):
        os.mkdir(data_source+'_GAN_results'+noise_factor_S)
    noise_img_path = data_source+'_GAN_results'+noise_factor_S+ '/initial_noise.png'
    # train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
    show_noise(train_set[0:100], noise_img_path, noise_factor, show=False)


    # networks : generator
    with tf.variable_scope('G'):
        z = tf.placeholder(tf.float32, shape=(None, n_sample_from))
        G_z = generator(z)

    # networks : discriminator
    with tf.variable_scope('D') as scope:
        drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
        x = tf.placeholder(tf.float32, shape=(None, 784))
        D_real = discriminator(x, drop_out)
        scope.reuse_variables()
        D_fake = discriminator(G_z, drop_out)


    # loss for each network
    eps = 1e-2
    D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
    G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

    # trainable variables for each network
    t_vars = tf.trainable_variables()
    D_vars = [var for var in t_vars if 'D_' in var.name]
    G_vars = [var for var in t_vars if 'G_' in var.name]

    # optimizer for each network
    D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

    # open session and initialize all variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # results save folder
    if not os.path.isdir(data_source+'_GAN_results'+noise_factor_S):
        os.mkdir(data_source+'_GAN_results'+noise_factor_S)
    if not os.path.isdir(data_source+'_GAN_results'+noise_factor_S+'/Random_results'):
        os.mkdir(data_source+'_GAN_results'+noise_factor_S+'/Random_results')
    if not os.path.isdir(data_source+'_GAN_results'+noise_factor_S+'/Fixed_results'):
        os.mkdir(data_source+'_GAN_results'+noise_factor_S+'/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # training-loop
    np.random.seed(int(time.time()))
    start_time = time.time()
    for epoch in range(train_epoch):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        for iter in range(train_set.shape[0] // batch_size):
            # update discriminator
            x_ = train_set[iter*batch_size:(iter+1)*batch_size]
            z_ = np.random.normal(0, 1, (batch_size, n_sample_from))

            loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: 0.3})
            D_losses.append(loss_d_)

            # update generator
            z_ = np.random.normal(0, 1, (batch_size, n_sample_from))
            loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: 0.3})
            G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))

        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        # generate data for calculating FID
        path_4_FID = data_source+'_GAN_results'+noise_factor_S+'/saved_result4FID'
        if (epoch+1) % train_epoch == 0:
            p = data_source+'_GAN_results' + noise_factor_S + '/Random_results/GAN_' + str(epoch + 1) + '.png'
            fixed_p = data_source+'_GAN_results' + noise_factor_S + '/Fixed_results/GAN_' + str(epoch + 1) + '.png'
            show_result(sess, G_z, z, drop_out, (epoch + 1), save=True, path=p, isFix=False)
            show_result(sess, G_z, z, drop_out, (epoch + 1), save=True, path=fixed_p, isFix=True)
            # save the result to calculate FID
            z_ = np.random.normal(0, 1, (n_4_FID, n_sample_from))
            test_images = sess.run(G_z, {z: z_, drop_out: 0.0})
            np.save(path_4_FID, test_images)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
    print("Training finish!... save training results")
    with open(data_source+'_GAN_results'+noise_factor_S+'/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)
    show_train_hist(train_hist, save=True, path=data_source+'_GAN_results'+noise_factor_S+'/GAN_train_hist.png')
# images = []
# for e in range(train_epoch):
#     img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

    sess.close()


if __name__ == "__main__":

    if len(sys.argv)>2:
        noise_factor = float(sys.argv[1])
    else:
        noise_factor = 0.2


    main(noise_factor)
