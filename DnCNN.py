import sys
sys.path.append('..')
import numpy as np
from skimage import io
import tensorflow as tf
from AWGN_create_train_test_val import get_AWGN_train_test_val


class denoising_autoencoder(object):
    def __init__(self):
        self.seed = np.random.seed(1234)
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = get_AWGN_train_test_val()

        self.train_x = self.train_x.astype(dtype='float')
        self.train_y = self.train_y.astype(dtype='float')

        self.test_x = self.test_x.astype(dtype='float')
        self.test_y = self.test_y.astype(dtype='float')

        self.val_x = self.val_x.astype(dtype='float')
        self.val_y = self.val_y.astype(dtype='float')

        self.img_dim = list(self.train_x.shape[1:])

        #place_holders
        self.x_ = None
        self.y_ = None

        #layers
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.MSE = None

        #tensorflow objects
        self.tf_session = None



    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_ = tf.placeholder('float', [None] + self.img_dim, name='noisy_image')
        y_ = tf.placeholder('float', [None] + self.img_dim, name='residual_noise')

        return x_, y_

    def create_model(self, x_batch, y_batch):
        """ 
        Build the denoiser
        """
        input_layer = tf.reshape(x_batch,[-1] + self.img_dim)

        conv1 = tf.layers.conv2d(inputs=input_layer,
                                filters=32,
                                kernel_size=[3, 3],
                                activation=tf.nn.relu,
                                padding='same',
                                name='conv1')

        conv2 = tf.layers.conv2d(inputs=conv1,
                                filters=64,
                                kernel_size=[3, 3],
                                padding='same',
                                activation=tf.nn.relu,
                                name='conv2')
    
        conv3 = tf.layers.conv2d(inputs=conv2,
                                filters=3,
                                kernel_size=[3, 3],
                                padding='same',
                                activation=None,
                                name='conv3')
        
        MSE = 1/2*tf.reduce_mean(tf.square(conv3 - y_batch))

        return conv1, conv2, conv3, MSE

    def PSNR(self):
        pass

    def run(self, epochs=10, batch_size=10):

        self.x_, self.y_ = self.create_placeholders()
        self.conv1, self.conv2, self.conv3, self.MSE = self.create_model(self.x_, self.y_)

        with tf.Session() as self.tf_session:

            train_step = tf.train.AdamOptimizer(1e-4).minimize(self.MSE)
            init_op = tf.global_variables_initializer()
            self.tf_session.run(init_op)

            for epoch in range(epochs):
                for i in np.arange(0,self.train_x.shape[0],batch_size):
                    if i%50 == 0:
                        print('Step %d' % i)
                    tr_feed = {self.y_: self.train_y[epoch:epoch+batch_size], self.x_: self.train_x[epoch:epoch+batch_size]}
                    train_step.run(feed_dict=tr_feed)

            noisy_image = self.test_x[0]
            noisy_image = np.expand_dims(noisy_image, axis=0)
            inf_feed = {self.x_: noisy_image}
            print self.tf_session.run([self.conv3], feed_dict=inf_feed)




if __name__ == '__main__':
    dae = denoising_autoencoder()
    dae.run()
    