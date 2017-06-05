import sys
import os
import numpy as np
from skimage import io
import glob

NOISE_LEVELS = [15, 25, 35, 45, 55]

display = 150
data_dir = '/mnt/hdd1/BSR/BSDS500/data/images'

def crop(image,offset):
    return image[offset:offset+180,offset:offset+180,:]

def make_dirs():
    if not os.path.exists(data_dir + '/AWGN/train'):
        os.makedirs(data_dir + '/AWGN/train')
    if not os.path.exists(data_dir + '/AWGN/test'):
        os.makedirs(data_dir + '/AWGN/test')
    if not os.path.exists(data_dir + '/AWGN/val'):
        os.makedirs(data_dir + '/AWGN/val')

def save_images(dir,x,y):
    i=0
    for _x, _y in zip(x.astype('uint8'), y.astype('uint8')):
        io.imsave(data_dir+'/AWGN/' + dir + '/' + str(i) + '_x' + '.jpg', _x)
        io.imsave(data_dir+'/AWGN/' + dir + '/' + str(i) + '_y' + '.jpg', _y)
        i = i + 1

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

    train_tensor = [train[i] for i in range(len(train))]
    test_tensor = [test[i] for i in range(len(test))]
    val_tensor = [val[i] for i in range(len(val))]

    for i, t in enumerate(train_tensor):
        train_tensor[i] = crop(t,100)
    for i, t in enumerate(test_tensor):
        test_tensor[i] = crop(t,100)
    for i, t in enumerate(val_tensor):
        val_tensor[i] = crop(t,100)

    train_tensor = np.array(train_tensor)
    test_tensor = np.array(test_tensor)
    val_tensor = np.array(val_tensor)
    clean_tensor = np.vstack((train_tensor, test_tensor, val_tensor)).astype(np.int16)
    np.random.shuffle(clean_tensor)
    noise_shape = [100] + list(clean_tensor.shape[1:])
    noise0 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[0], size=noise_shape)
    noise1 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[1], size=noise_shape)
    noise2 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[2], size=noise_shape)
    noise3 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[3], size=noise_shape)
    noise4 = np.random.normal(loc=0.0, scale=NOISE_LEVELS[4], size=noise_shape)
    noise = np.vstack((noise0, noise1, noise2, noise3, noise4))
    np.random.shuffle(noise)
    noise = np.rint(noise)

    noisy_tensor = clean_tensor + noise
    clean_tensor = clean_tensor.astype(np.uint8)
    noisy_tensor = np.clip(noisy_tensor, 0, 255).astype(np.uint8)


    if residual_learning == True:
        train_x = noisy_tensor[0:300, :, :, :]
        train_y = noise[0:300, :, :, :]

        val_x = noisy_tensor[300:400, :, :, :]
        val_y = noise[300:400, :, :, :]

        test_x = noisy_tensor[400:, :, :, :]
        test_y = noise[400:, :, :, :]

    else:
        train_x = noisy_tensor[0:300, :, :, :]
        train_y = clean_tensor[0:300, :, :, :]

        val_x = noisy_tensor[300:400, :, :, :]
        val_y = clean_tensor[300:400, :, :, :]

        test_x = noisy_tensor[400:, :, :, :]
        test_y = clean_tensor[400:, :, :, :]

    save_images('train', train_x, train_y)
    save_images('test', test_x, test_y)
    save_images('val', val_x, val_y)
    

    return train_x, train_y, val_x, val_y, test_x, test_y


def get_AWGN_train_test_val():
    train_x_path = sorted(glob.glob(data_dir + '/AWGN/train/*x.jpg'))
    train_y_path = sorted(glob.glob(data_dir + '/AWGN/train/*y.jpg'))
    val_x_path = sorted(glob.glob(data_dir + '/AWGN/val/*x.jpg'))
    val_y_path = sorted(glob.glob(data_dir + '/AWGN/val/*y.jpg'))
    test_x_path = sorted(glob.glob(data_dir + '/AWGN/test/*x.jpg'))
    test_y_path = sorted(glob.glob(data_dir + '/AWGN/test/*y.jpg'))

    train_x = io.imread_collection(train_x_path)
    train_y = io.imread_collection(train_y_path)
    val_x = io.imread_collection(val_x_path)
    val_y = io.imread_collection(val_y_path)
    test_x = io.imread_collection(test_x_path)
    test_y = io.imread_collection(test_y_path)

    train_x = img_collection_to_numpy(train_x)
    train_y = img_collection_to_numpy(train_y)
    val_x = img_collection_to_numpy(val_x)
    val_y = img_collection_to_numpy(val_y)
    test_x = img_collection_to_numpy(test_x)
    test_y = img_collection_to_numpy(test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':
    create_AWGN_train_test_val()
    train_x, train_y, val_x, val_y, test_x, test_y = get_AWGN_train_test_val()
    io.imshow_collection([train_x[display], train_y[display]])
    MSE = np.mean(np.square(train_x[display]-train_y[display]))
    PSNR = 10*np.log10((255**2)/MSE)
    print(PSNR)
    io.show()
