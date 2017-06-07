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


def conv2d_with_BN(x, kernel_size, input_channels, output_channels, layer_name, BN=True, act=True):
    W = weight_variable([kernel_size, kernel_size, input_channels, output_channels], 'W_' + layer_name)
    b = bias_variable([output_channels], 'b_' + layer_name)
    o = offset_variable([x.get_shape().as_list()[1], x.get_shape().as_list()[1], output_channels], 'o_' + layer_name)
    s = scale_variable([x.get_shape().as_list()[1], x.get_shape().as_list()[1], output_channels], 's_' + layer_name)

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            variable_summaries(W)
        with tf.name_scope('biases'):
            variable_summaries(b)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
            tf.summary.histogram('pre_activations', preactivate)
        if BN == True and act==True:
            with tf.name_scope('BN_Wx_plus_b'):
                batch_mean1, batch_var1 = tf.nn.moments(preactivate,[0])
                BN_preactivate = tf.nn.batch_normalization(preactivate, batch_mean1, batch_var1, o, s, 0.01)
                tf.summary.histogram('BN_pre_activations', BN_preactivate)
                activations = tf.nn.relu(BN_preactivate, name='activation')
        elif BN == False and act == True:
            activations = tf.nn.relu(preactivate, name='activation')
        elif BN == False and act == False:
            activations = tf.identity(preactivate, name='activation')   
        tf.summary.histogram('activations', activations)

    return activations



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

        self.save_path = '/home/kenleejr92/Denoising/tmp/DnCNN'


    def create_placeholders(self):
        """ 
        Create the TensorFlow placeholders for the model.
        """
        x_ = tf.placeholder('float', [None] + self.img_dim, name='noisy_image')
        y_ = tf.placeholder('float', [None] + self.img_dim, name='residual_noise')

        return x_, y_


    def create_model(self, x_batch, y_batch, num_layers=19):
        """ 
        Build the denoiser
        """
        input_layer = tf.reshape(x_batch,[-1] + self.img_dim)

        h_conv0 = conv2d_with_BN(input_layer, kernel_size=3, input_channels=3, output_channels=32, layer_name='conv0', BN=False, act=True)
        tf.add_to_collection('h_conv0', h_conv0)
        layer_array = [h_conv0]
        for i in np.arange(1, num_layers-2):
            h_conv = conv2d_with_BN(layer_array[i-1], 3, layer_array[i-1].get_shape().as_list()[3], 64, 'h_conv' + str(i))
            tf.add_to_collection('h_conv' + str(i), h_conv)
            layer_array.append(h_conv)

        output = conv2d_with_BN(layer_array[-1], 3, layer_array[i-1].get_shape().as_list()[3], 3, 'h_conv' + str(num_layers), BN=False, act=False)
        MSE = tf.reduce_mean(tf.square(output - y_batch))
        with tf.name_scope('metrics'):
            tf.summary.scalar('MSE', MSE)
            variable_summaries(output)

        PSNR = (255**2)/MSE

        tf.add_to_collection('output', output)
        tf.add_to_collection('MSE', MSE)

        return output, MSE, PSNR


    def restore_model(self, tf_session, global_step=50):
        new_saver = tf.train.import_meta_graph(self.save_path + '-' + str(global_step) + '.meta')
        new_saver.restore(tf_session, tf.train.latest_checkpoint('/home/kenleejr92/Denoising/tmp/'))

        self.output = tf.get_collection('output')
        self.MSE = tf.get_collection('MSE')


    def train(self, epochs=50, batch_size=128):
        """
        Training
        """
        x_, y_ = self.create_placeholders()
        output, MSE, PSNR = self.create_model(x_, y_)
        tf_saver = tf.train.Saver()
        tf_session = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/home/kenleejr92/Denoising/tmp/train', tf_session.graph)
        val_writer = tf.summary.FileWriter('/home/kenleejr92/Denoising/tmp/val', tf_session.graph)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(MSE)
        init_op = tf.global_variables_initializer()
        tf_session.run(init_op)

        for epoch in range(epochs+1):
            if epoch%5 == 0:
                print('Step %d' % epoch)
                tr_feed = {y_: self.train_y[epoch:epoch+batch_size], x_: self.train_x[epoch:epoch+batch_size]}
                print tf_session.run(MSE, feed_dict = tr_feed)
                tf_saver.save(tf_session, self.save_path, global_step=epoch)
                summary = tf_session.run(merged, feed_dict=tr_feed)
                train_writer.add_summary(summary, epoch)

            for i in np.arange(0,self.train_x.shape[0],batch_size):
                tr_feed = {y_: self.train_y[epoch:epoch+batch_size], x_: self.train_x[epoch:epoch+batch_size]}
                tf_session.run(train_step, feed_dict=tr_feed)


    def inference(self):
        tf_session = tf.Session()
        self.restore_model(tf_session)
        inf_feed = {'noisy_image:0': self.test_x[10:20]}
        residual_noise =  tf_session.run(self.output, feed_dict=inf_feed)
        clean_images = np.clip(np.squeeze(self.test_x[10:20] - residual_noise), 0, 255).astype(np.uint8)
        original_img = np.clip(np.squeeze(self.test_x[10:20] - self.test_y[10:20]), 0, 255).astype(np.uint8)
        for i in range(10):
            io.imshow_collection([self.test_x[i+10], clean_images[i], original_img[i]])
            io.show()

    def denoise_img(self, image):
        tf_session = tf.Session()
        self.restore_model(tf_session)
        width = image.shape[0]
        height = image.shape[1]
        patches = np.zeros((width, height, 3))
        for h in np.arange(0, height-50, 50):
            for w in np.arange(0, width-50, 50):
                patch = np.expand_dims(image[h:h+50, w:w+50, :], axis=0)
                if patch.shape != (1, 50, 50, 3): continue
                inf_dict = {'noisy_image:0': patch}
                patches[h:h+50, w:w+50, :] = patch.astype(np.int16) - np.squeeze(tf_session.run(self.output, feed_dict=inf_dict))
            
        return patches
        


if __name__ == '__main__':
    dae = denoising_autoencoder()
    # dae.train()
    # dae.inference()
    clean = io.imread('/mnt/hdd1/BSR/BSDS500/data/images/test/226022.jpg')
    noise = np.rint(np.random.normal(loc=0.0, scale=55.0, size=clean.shape))
    noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    denoised = dae.denoise_img(noisy)
    noisy = np.clip(noisy, 0, 255)
    denoised = np.rint(np.clip(denoised, 0, 255)).astype(np.uint8)
    io.imshow_collection([clean, noisy, denoised])
    io.show()

    