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

        self.img_dim = list(self.train_x.shape[1:])

        #place_holders
        self.x_clean = None
        self.x_noisy = None

        #layers
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.MSE = None

        #tensorflow objectts
        self.tf_session = None



    def _create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_clean = tf.placeholder('float', [None] + self.img_dim, name='x_clean')
        x_noisy = tf.placeholder('float', [None] + self.img_dim, name='x_noisy')

        return x_clean, x_noisy

    def _create_model(self, x_batch, y_batch):
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
        
        loss = 1/2*tf.reduce_mean(tf.square(conv3 - y_batch))

        return conv1, conv2, conv3, loss

    def run(self):
        print self.train_x.shape
        print self.train_y.shape

        self.x_clean, self.x_noisy = self._create_placeholders()
        self.conv1, self.conv2, self.conv3, self.MSE = self._create_model(self.x_noisy, self.x_clean)

        with tf.Session() as self.tf_session:
            init_op = tf.global_variables_initializer()
            self.tf_session.run(init_op)
            tr_feed = {self.x_clean: self.train_y[0:5, :, :, :], self.x_noisy: self.train_x[0:5, :, :, :]}
            print self.tf_session.run([self.MSE], feed_dict=tr_feed)
            # train_step.run(tr_feed)



if __name__ == '__main__':
    dae = denoising_autoencoder()
    dae.run()