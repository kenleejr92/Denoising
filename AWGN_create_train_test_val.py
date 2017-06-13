import sys
import os
import numpy as np
from skimage import io
import glob
import cPickle as pickle

NOISE_LEVELS = [15, 15, 15, 15, 15]

display = 5
data_dir = '/mnt/hdd1/BSR/BSDS500/data/images'

def crop(image, size=50):
    width = image.shape[0]
    height = image.shape[1]

    h = np.random.randint(low=0, high=height-50)
    w = np.random.randint(low=0, high=width-50)
    return image[w:w+50, h:h+50, :]

def make_dirs():
    if not os.path.exists(data_dir + '/AWGN/train'):
        os.makedirs(data_dir + '/AWGN/train')
    if not os.path.exists(data_dir + '/AWGN/test'):
        os.makedirs(data_dir + '/AWGN/test')
    if not os.path.exists(data_dir + '/AWGN/val'):
        os.makedirs(data_dir + '/AWGN/val')

# def save_images(dir, x, id):
#     i=0
#     for _x in x:
#         io.imsave(data_dir+'/AWGN/' + dir + '/' + str(i).zfill(4) + '_' + id + '.jpg', _x)
#         i = i + 1

def save_images(dir, y, id):
    with open(data_dir+'/AWGN/' + dir + '/' + id + '.pkl', 'w+') as f:
        pickle.dump(y, f)

def sample_from_image(image, num_samples=20, size=50):
    samples = np.empty((1, size, size, 3), np.uint8)
    for i in range(num_samples):
        cropped = crop(image, size)
        cropped = np.expand_dims(cropped, axis=0)
        samples = np.vstack((samples, cropped))
    return samples[1:, :, :, :]

def read_pickle(file):
    with open(file, 'r') as f:
        return pickle.load(f)

def img_collection_to_numpy(img_collection):
    tensor = [img_collection[i] for i in range(len(img_collection))]
    return np.array(tensor)

def create_AWGN_train_test_val():
    '''
    Adds Gaussian noise from 5 different NOISE_LEVELS to the BSD dataset
    Creates train, val, and test images
    '''
    np.random.seed(1234)
    num_samples = 800
    num_images = 500

    make_dirs()

    train = io.imread_collection(data_dir + '/train/*.jpg')
    val = io.imread_collection(data_dir + '/val/*.jpg')
    test = io.imread_collection(data_dir + '/test/*.jpg')

    train_arr = [train[i] for i in range(len(train))]
    test_arr = [test[i] for i in range(len(test))]
    val_arr = [val[i] for i in range(len(val))]

    train_tensor = np.zeros((num_samples*len(train_arr), 50, 50, 3))
    test_tensor = np.zeros((num_samples*len(test_arr), 50, 50, 3))
    val_tensor = np.zeros((num_samples*len(val_arr), 50, 50, 3))

    for i, t in enumerate(train_arr):
        train_tensor[i*num_samples:i*num_samples+num_samples, :, :, :] = sample_from_image(t, num_samples=num_samples)
    for i, t in enumerate(test_arr):
        test_tensor[i*num_samples:i*num_samples+num_samples, :, :, :] = sample_from_image(t, num_samples=num_samples)
    for i, t in enumerate(val_arr):
        val_tensor[i*num_samples:i*num_samples+num_samples, :, :, :] = sample_from_image(t, num_samples=num_samples)

    train_tensor = np.array(train_tensor)
    test_tensor = np.array(test_tensor)
    val_tensor = np.array(val_tensor)
    clean_tensor = np.vstack((train_tensor, test_tensor, val_tensor)).astype(np.int16)
    np.random.shuffle(clean_tensor)
    noise_shape = [2000] + list(clean_tensor.shape[1:])
    noise0 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[0], size=noise_shape)
    noise1 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[1], size=noise_shape)
    noise2 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[2], size=noise_shape)
    noise3 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[3], size=noise_shape)
    noise4 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[4], size=noise_shape)
    noise = np.vstack((noise0, noise1, noise2, noise3, noise4))
    np.random.shuffle(noise)
    # Round appropriate?
    noise = np.rint(noise)

    noisy_tensor = clean_tensor + noise
    noisy_tensor = np.clip(noisy_tensor, 0, 255).astype(np.uint8)
    noise = noisy_tensor.astype(np.int16) - clean_tensor.astype(np.int16)
    clean_tensor = clean_tensor.astype(np.uint8)
    # io.imshow_collection([clean_tensor[5], noisy_tensor[5], np.clip(noisy_tensor[5].astype(np.int16) - noise[5], 0, 255).astype(np.uint8)])
    # io.show()
    
    train_x = noisy_tensor[0:8000, :, :, :]
    train_y = noise[0:8000, :, :, :]

    val_x = noisy_tensor[8000:9000, :, :, :]
    val_y = noise[8000:9000, :, :, :]

    test_x = noisy_tensor[9000:, :, :, :]
    test_y = noise[9000:, :, :, :]

    train_c = clean_tensor[0:8000, :, :, :]
    val_c = clean_tensor[8000:9000, :, :, :]
    test_c = clean_tensor[9000:, :, :, :]

    save_images('train', train_x, id='x')
    save_images('train', train_c, id='c')
    save_images('train', train_y, id='y')

    save_images('test', test_x, id='x')
    save_images('test', test_c, id='c')
    save_images('test', test_y, id='y')

    save_images('val', val_x, id='x')
    save_images('val', val_c, id='c')
    save_images('val', val_y, id='y')


    return train_x, train_y, val_x, val_y, test_x, test_y


def get_AWGN_train_test_val(return_clean_imgs = True):
    # train_x_path = sorted(glob.glob(data_dir + '/AWGN/train/*_x.jpg'))
    # val_x_path = sorted(glob.glob(data_dir + '/AWGN/val/*_x.jpg'))
    # test_x_path = sorted(glob.glob(data_dir + '/AWGN/test/*_x.jpg'))

    # train_x = io.imread_collection(train_x_path)
    # val_x = io.imread_collection(val_x_path)
    # test_x = io.imread_collection(test_x_path)
    # train_x = img_collection_to_numpy(train_x)
    # val_x = img_collection_to_numpy(val_x)
    # test_x = img_collection_to_numpy(test_x)

    train_x = read_pickle(data_dir + '/AWGN/train/x.pkl')
    val_x = read_pickle(data_dir + '/AWGN/val/x.pkl')
    test_x = read_pickle(data_dir + '/AWGN/test/x.pkl')

    train_y = read_pickle(data_dir + '/AWGN/train/y.pkl')
    val_y = read_pickle(data_dir + '/AWGN/val/y.pkl')
    test_y = read_pickle(data_dir + '/AWGN/test/y.pkl')

    if return_clean_imgs == True:
        train_c = read_pickle(data_dir + '/AWGN/train/c.pkl')
        val_c = read_pickle(data_dir + '/AWGN/val/c.pkl')
        test_c = read_pickle(data_dir + '/AWGN/test/c.pkl')

        # train_c_path = sorted(glob.glob(data_dir + '/AWGN/train/*_c.jpg'))
        # val_c_path = sorted(glob.glob(data_dir + '/AWGN/val/*_c.jpg'))
        # test_c_path = sorted(glob.glob(data_dir + '/AWGN/test/*_c.jpg'))

        # train_c = io.imread_collection(train_c_path)
        # val_c = io.imread_collection(val_c_path)
        # test_c = io.imread_collection(test_c_path)
        # train_c = img_collection_to_numpy(train_c)
        # val_c = img_collection_to_numpy(val_c)
        # test_c = img_collection_to_numpy(test_c)

        return train_x, train_y, train_c, val_x, val_y, val_c, test_x, test_y, test_c

    else:
        return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':
    create_AWGN_train_test_val()
    train_x, train_y, train_c, val_x, val_y, val_c, test_x, test_y, test_c = get_AWGN_train_test_val()
    io.imshow_collection([train_c[display], train_x[display], np.clip(train_x[display].astype(np.int16) - train_y[display], 0, 255).astype(np.uint8)])
    MSE = np.mean(np.square(train_y[display]))
    PSNR = 10*np.log10((255**2)/MSE)
    print(PSNR)
    io.show()
