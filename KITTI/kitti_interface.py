import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple
import matplotlib.image as mpimg
import numpy as np
from skimage import io
from skimage.color import grey2rgb
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = '/mnt/hdd1/KITTI/training'
CALIB_DIR = BASE_DIR + '/calib'
VELO_DIR = BASE_DIR + '/velodyne'
LEFTIMG_DIR = BASE_DIR + '/image_2'

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line == '\n': break
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def get_images(imfiles, imformat):
    """Generator to read image files."""
    for file in imfiles:
        # Convert to uint8 and BGR for OpenCV if requested
        if imformat is 'cv2':
            im = np.uint8(mpimg.imread(file) * 255)

            # Convert RGB to BGR
            if len(im.shape) > 2:
                im = im[:, :, ::-1]
        else:
            im = mpimg.imread(file)

        yield im


def get_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for filename in velo_files:
        scan = np.fromfile(filename, dtype=np.float32)

    return scan.reshape((-1, 4))



if __name__ == '__main__':
    file = '/005965'
    data = read_calib_file(CALIB_DIR + file + '.txt')
    velo = get_velo_scans([VELO_DIR + file + '.bin'])
    img = io.imread(LEFTIMG_DIR + file + '.png')
    P2 = np.reshape(data['P2'], (3, 4))
    P3 = np.reshape(data['P3'], (3, 4))
    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = np.reshape(data['R0_rect'], (3, 3))
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, 0:4] = np.reshape(data['Tr_velo_to_cam'], (3, 4))
    #coords is the projection onto the left image
    coords = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(np.transpose(velo)))).T
    
    #This was my best guess on how to scale the coordinates
    image_size = img.shape
    scalerx = MinMaxScaler(feature_range=(0, image_size[0]-1))
    scalery = MinMaxScaler(feature_range=(0, image_size[1]-1))
    scaleri = MinMaxScaler(feature_range=(-1.0, 1.0))
    coords[:, 0] = scalerx.fit_transform(coords[:, 0])
    coords[:, 1] = scalery.fit_transform(coords[:, 1])
    coords[:, 2] = scaleri.fit_transform(coords[:, 2])

    image = np.zeros(image_size)
    for x, y, i in coords:
        image[int(x), int(y)] = i

    #plot 3-D point cloud
    f2 = plt.figure()
    ax2 = f2.add_subplot(111, projection='3d')
    velo_range = range(0, velo.shape[0], 100)
    ax2.scatter(velo[velo_range, 0],
            velo[velo_range, 1],
            velo[velo_range, 2],
            c=velo[velo_range, 3],
            cmap='gray')
    ax2.set_title('Velodyne scan (subsampled)')

    plt.show()

    #plot 2-Dimage
    io.imshow_collection([image, img])
    io.show()