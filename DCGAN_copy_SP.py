import sys, os, time, itertools, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def corrupt(X,corNum=10):
    N,p = X.shape[0],X.shape[1]
    for i in range(N):
        loclist = np.random.randint(0, p, size = corNum)
        for j in loclist:
            if X[i,j,0] > 0.5:
                X[i,j, 0] = 0
            else:
                X[i,j, 0] = 1
    return X
# G(z)
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)

        return o


def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)

        return o, conv5

digit_n = 10
n_sample_from = 100
h = w = 64
np.random.seed(595)
fixed_z_ = np.random.normal(0, 1, (digit_n*digit_n, n_sample_from))




def show_result(sess, G_z, z, drop_out, num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    np.random.seed(595)

    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

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
            I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = test_images[k].reshape(64, 64)

    plt.figure(figsize=(8, 8))
    plt.imshow(I_generated, cmap='gray')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_noise(test_images, path, noise_factor, show=False):



    I_generated = np.empty((h * digit_n, w * digit_n))
    for i in range(digit_n):
        for j in range(digit_n):
            k = i*10 + j
            I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = test_images[k].reshape(64, 64)

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
    train_epoch = 1
    n_4_FID = 10
    noise_factor_S = 'DCGAN_SP' + str(noise_factor)
    np.random.seed(595)

    data_source = 'MNIST'


    if data_source == 'MNIST':
        data_dir = 'input_data/MNIST_data/'
        mnist = input_data.read_data_sets(data_dir, one_hot=True, reshape=[])


    # variables : input
    x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
    z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
    isTrain = tf.placeholder(dtype=tf.bool)

    # networks : generator
    G_z = generator(z, isTrain)

    # networks : discriminator
    D_real, D_real_logits = discriminator(x, isTrain)
    D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

    # loss for each network
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

    # trainable variables for each network
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]

    # optimizer for each network
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
        G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

    # open session and initialize all variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # MNIST resize and normalization
    train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()

    train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

    train_set = corrupt(train_set, corNum=int(noise_factor * 4096))


    # plot image after adding noise
    if not os.path.isdir(data_source+'_GAN_results'+noise_factor_S):
        os.mkdir(data_source+'_GAN_results'+noise_factor_S)
    noise_img_path = data_source+'_GAN_results'+noise_factor_S+ '/initial_noise.png'
    # train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
    show_noise(train_set[0:100], noise_img_path, noise_factor, show=False)


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

        for iter in range(mnist.train.num_examples // batch_size):
            # update discriminator
            x_ = train_set[iter * batch_size:(iter + 1) * batch_size]
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

            loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
            D_losses.append(loss_d_)

            # update generator
            z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
            loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
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
            show_result((epoch + 1), save=True, path=p, isFix=False)
            show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
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

