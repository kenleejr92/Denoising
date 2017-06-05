import sys
sys.path.append('..')
import numpy as np
from skimage import io
import tensorflow as tf
from AWGN_create_train_test_val import get_AWGN_train_test_val


def weight_variable(shape, Name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=Name)

def bias_variable(shape, Name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=Name)

def offset_variable(shape, Name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name=Name)

def scale_variable(shape, Name):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial, name=Name)

def conv2d(x, W, b, layer_name, act=tf.nn.relu,):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            variable_summaries(W)
        with tf.name_scope('biases'):
            variable_summaries(b)
        with tf.name_scope('W_plus_b'):
            preactivate = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
            tf.summary.histogram('pre_activations', preactivate)
        if act == None:
            activations = tf.identity(preactivate, name='activation')
        else: 
            activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

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

        #tf variables
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None


        #tf ops on the variables
        self.MSE = None
        self.PSNR = None
        self.h_conv1 = None
        self.h_conv2 = None
        self.output = None

        #tensorflow objects
        self.tf_session = None
        self.tf_saver = None
        self.save_path = '/home/kenleejr92/Denoising/tmp/DnCNN'


    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_ = tf.placeholder('float', [None] + self.img_dim, name='noisy_image')
        y_ = tf.placeholder('float', [None] + self.img_dim, name='residual_noise')

        return x_, y_

    def create_variables(self):
        W1 = weight_variable([3, 3, 3, 32], 'W1')
        b1 = bias_variable([32], 'b1')

        o1 = offset_variable([3, 3, 3, 32], 'o1')
        s1 = scale_variable([3, 3, 3, 32], 's1')

        W2 = weight_variable([3, 3, 32, 64], 'W2')
        b2 = bias_variable([64], 'b2')

        o2 = offset_variable([3, 3, 32, 64], 'o1')
        s2 = scale_variable([3, 3, 32, 64], 's1')

        W3 = weight_variable([3, 3, 64, 3], 'W3')
        b3 = bias_variable([3], 'b3')

        o1 = offset_variable([3, 3, 3, 32], 'o1')
        s1 = scale_variable([3, 3, 3, 32], 's1')

        return W1, b1, W2, b2, W3, b3


    def create_model(self, x_batch, y_batch):
        """ 
        Build the denoiser
        """
        input_layer = tf.reshape(x_batch,[-1] + self.img_dim)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.create_variables()

        self.h_conv1 = conv2d(input_layer, self.W1, self.b1, 'conv1')
        self.h_conv2 = conv2d(self.h_conv1, self.W2, self.b2, 'conv2')
        self.output = conv2d(self.h_conv2, self.W3, self.b3, 'conv3', act=None)

        self.MSE = tf.reduce_mean(tf.square(self.output - y_batch))
        with tf.name_scope('MSE'):
            tf.summary.scalar('MSE',self.MSE)
        self.PSNR = (255**2)/self.MSE

        tf.add_to_collection('h_conv1', self.h_conv1)
        tf.add_to_collection('h_conv2', self.h_conv2)
        tf.add_to_collection('output', self.output)
        tf.add_to_collection('MSE', self.MSE)


    def restore_model(self, global_step=5):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.create_variables()
        new_saver = tf.train.import_meta_graph(self.save_path + '-' + str(global_step) + '.meta')
        new_saver.restore(self.tf_session, tf.train.latest_checkpoint('/home/kenleejr92/Denoising/tmp/'))

        self.h_conv1 = tf.get_collection('h_conv1')
        self.h_conv2 = tf.get_collection('h_conv2')
        self.output = tf.get_collection('output')
        self.MSE = tf.get_collection('MSE')


    def train(self, epochs=100, batch_size=10):
        """
        Training
        """
        self.x_, self.y_ = self.create_placeholders()
        self.create_model(self.x_, self.y_)
        self.tf_saver = tf.train.Saver([self.W1, self.W2, self.W3, self.b1, self.b2, self.b3])
        self.tf_session = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/home/kenleejr92/Denoising/tmp/train', self.tf_session.graph)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.MSE)
        init_op = tf.global_variables_initializer()
        self.tf_session.run(init_op)

        for epoch in range(epochs):
            if epoch%10 == 0:
                print('Step %d' % epoch)
                tr_feed = {self.y_: self.train_y[epoch:epoch+batch_size], self.x_: self.train_x[epoch:epoch+batch_size]}
                print 10*np.log10(self.tf_session.run(self.PSNR, feed_dict = tr_feed))
                self.tf_saver.save(self.tf_session, self.save_path, global_step=epoch)
                summary = self.tf_session.run(merged, feed_dict=tr_feed)
                train_writer.add_summary(summary, epoch)

            for i in np.arange(0,self.train_x.shape[0],batch_size):
                tr_feed = {self.y_: self.train_y[epoch:epoch+batch_size], self.x_: self.train_x[epoch:epoch+batch_size]}
                self.tf_session.run(train_step, feed_dict=tr_feed)


    def inference(self):
        self.tf_session = tf.Session()
        self.restore_model()
        inf_feed = {'noisy_image:0': self.test_x[0:5]}
        residual_noise =  self.tf_session.run(self.output, feed_dict=inf_feed)
        clean_images = np.squeeze(self.test_x[0:5] - residual_noise)

        io.imshow(clean_images[0])
        io.show()




if __name__ == '__main__':
    dae = denoising_autoencoder()
    dae.train()
    #dae.inference()
    