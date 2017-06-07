import sys
import os
import numpy as np
from skimage import io
import glob
import cPickle as pickle

NOISE_LEVELS = [15, 25, 35, 45, 55]

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

def save_images(dir, x):
    i=0
    for _x in x:
        io.imsave(data_dir+'/AWGN/' + dir + '/' + str(i) + '_x' + '.jpg', _x)
        i = i + 1

def save_noise(dir, y):
    with open(data_dir+'/AWGN/' + dir + '/' + 'y' + '.pkl', 'w+') as f:
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

def create_AWGN_train_test_val(residual_learning=True):
    '''
    Adds Gaussian noise from 5 different NOISE_LEVELS to the BSD dataset
    Creates train, val, and test images
    '''
    np.random.seed(1234)

    make_dirs()

    train = io.imread_collection(data_dir + '/train/*.jpg')
    val = io.imread_collection(data_dir + '/val/*.jpg')
    test = io.imread_collection(data_dir + '/test/*.jpg')

    train_arr = [train[i] for i in range(len(train))]
    test_arr = [test[i] for i in range(len(test))]
    val_arr = [val[i] for i in range(len(val))]

    train_tensor = np.zeros((20*len(train_arr), 50, 50, 3))
    test_tensor = np.zeros((20*len(test_arr), 50, 50, 3))
    val_tensor = np.zeros((20*len(val_arr), 50, 50, 3))

    for i, t in enumerate(train_arr):
        train_tensor[i*20:i*20+20, :, :, :] = sample_from_image(t)
        # if (i*20)%100 == 0:
        #     io.imshow(train_tensor[i*20])
        #     io.show()
    for i, t in enumerate(test_arr):
        test_tensor[i*20:i*20+20, :, :, :] = sample_from_image(t)
    for i, t in enumerate(val_arr):
        val_tensor[i*20:i*20+20, :, :, :] = sample_from_image(t)

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
    clean_tensor = clean_tensor.astype(np.uint8)
    noisy_tensor = np.clip(noisy_tensor, 0, 255).astype(np.uint8)


    if residual_learning == True:
        train_x = noisy_tensor[0:8000, :, :, :]
        train_y = noise[0:8000, :, :, :]

        val_x = noisy_tensor[8000:9000, :, :, :]
        val_y = noise[8000:9000, :, :, :]

        test_x = noisy_tensor[9000:, :, :, :]
        test_y = noise[9000:, :, :, :]

    else:
        train_x = noisy_tensor[0:8000, :, :, :]
        train_y = clean_tensor[0:8000, :, :, :]

        val_x = noisy_tensor[8000:9000, :, :, :]
        val_y = clean_tensor[8000:9000, :, :, :]

        test_x = noisy_tensor[9000:, :, :, :]
        test_y = clean_tensor[9000:, :, :, :]

    save_images('train', train_x)
    save_noise('train', train_y)
    save_images('test', test_x)
    save_noise('test', test_y)
    save_images('val', val_x)
    save_noise('val', val_y)
    

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_AWGN_train_test_val():
    train_x_path = sorted(glob.glob(data_dir + '/AWGN/train/*x.jpg'))
    val_x_path = sorted(glob.glob(data_dir + '/AWGN/val/*x.jpg'))
    test_x_path = sorted(glob.glob(data_dir + '/AWGN/test/*x.jpg'))

    train_x = io.imread_collection(train_x_path)
    train_y = read_pickle(data_dir + '/AWGN/train/y.pkl')
    val_x = io.imread_collection(val_x_path)
    val_y = read_pickle(data_dir + '/AWGN/val/y.pkl')
    test_x = io.imread_collection(test_x_path)
    test_y = read_pickle(data_dir + '/AWGN/test/y.pkl')

    train_x = img_collection_to_numpy(train_x)
    val_x = img_collection_to_numpy(val_x)
    test_x = img_collection_to_numpy(test_x)

    return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':
    create_AWGN_train_test_val()
    train_x, train_y, val_x, val_y, test_x, test_y = get_AWGN_train_test_val()
    io.imshow_collection([train_x[display], train_y[display]])
    MSE = np.mean(np.square(train_x[display]-train_y[display]))
    PSNR = 10*np.log10((255**2)/MSE)
    print(PSNR)
    io.show()
