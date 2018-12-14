import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
#plt.switch_backend('agg')
import os
import time

def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))
class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list=[784,400]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = np.sqrt(6. / (self.dim_list[i] + self.dim_list[i+1]))
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],
                                                             np.negative(init_max_value),init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
        ## Decoders parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))
        ## Placeholder for input
        self.input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## coding graph :
        last_layer = self.input_x
        for weight,bias in zip(self.W_list,self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            last_layer = hidden
        self.hidden = hidden 
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer
   
        self.cost = tf.reduce_mean(tf.square(self.input_x - self.recon))
        #self.cost = tf.losses.log_loss(self.recon, self.input_x)
        self.train_step = tf.train.AdamOptimizer().minimize(self.cost)
        sess.run(tf.global_variables_initializer())

    def fit(self, X, sess, path = "", file_name = "", learning_rate=0.15,
            iteration=200, batch_size=50, init=False,verbose=False):
        
        assert X.shape[1] == self.dim_list[0]
        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]
        ls_loss = []
        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                _,loss = sess.run([self.train_step,self.cost],feed_dict = {self.input_x:X[one_batch]})
            ls_loss.append(loss)
            if verbose and i%20==0:
                e = self.cost.eval(session = sess,feed_dict = {self.input_x: X[one_batch]})
                print ("    iteration : ", i ,", cost : ", e)
                
        np.save(path+file_name,np.array(ls_loss))
                
    def plot(self, sess, num_gen=10, path="", fig_name=""):
        """
        FLAG_gen: flag of generation or reconstruction. True = generation
        x: reconstruction input
        num_gen: number of generation
        path: path of saving
        fig_name: name of saving
        """
        
        if not os.path.exists(path):
            os.mkdir(path)
            
        h = w = 28
        np.random.seed(595)
        z = np.random.normal(size=[num_gen, self.dim_list[-1]])
        rvae_generated = self.generator(z,sess)
        
        # plot of reconstruction
        n = np.sqrt(num_gen).astype(np.int32)
        I_generated = np.empty((h*n, w*n))
        for i in range(n):
            for j in range(n):
                I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = rvae_generated[i*n+j, :].reshape(28, 28)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(I_generated, cmap='gray')
        plt.savefig(path+fig_name)
#        plt.show()
        
    def transform(self, X, sess):
        return self.hidden.eval(session = sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session = sess,feed_dict={self.input_x: X})
    
    # z -> x
    def generator(self, z, sess):
        x_hat = sess.run(self.recon, feed_dict={self.hidden: z})
        return x_hat
    
def test():
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:1000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 1000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:10000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 10,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:20000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 20,000 Runing time:" + str(time.time() - start_time) + " s")

    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,625,400,225,100])
        error = ae.fit(x[:50000] ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 1000, verbose=False)

    print ("size 50,000 Runing time:" + str(time.time() - start_time) + " s")
if __name__ == "__main__":
    
    os.chdir("../../")
    x = np.load(r"./data/data.npk")
    start_time = time.time()
    with tf.Session() as sess:
        ae = Deep_Autoencoder(sess = sess, input_dim_list=[784,225,100])
        error = ae.fit(x ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 500, verbose=True)
        R = ae.getRecon(x, sess = sess)
        print ("size 100 Runing time:" + str(time.time() - start_time) + " s")
        error = ae.fit(R ,sess = sess, learning_rate=0.01, batch_size = 500, iteration = 500, verbose=True)
    #test()
