import numpy as np
import skimage
import imagesize
from itertools import cycle
from sklearn.utils import shuffle
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
from utils import dir_listing, file_listing, take


def read_image(path, size):
    '''
    Read image from specified path and resize it if size specified

    Args:
        path (string): Path to image
        size (tuple): Size of output image, tuple of integers in format (W, H)

    Returns:
        Numpy array representing image with values in range (0, 1)
        or None if image is too small (both sides are smaller than target size)
    '''
    try:
        img_np = skimage.io.imread(path)
        if img_np.shape[0] < size[0] or img_np.shape[1] < size[1]:
            return None
        img_np = skimage.transform.resize(img_np, size)
        if img_np.ndim == 2 or img_np.shape[2] == 1:
            img_np = np.reshape(img_np, (img_np.shape[0], img_np.shape[1], 1))
            img_np = np.repeat(img_np, 3, axis=-1)
        return img_np
    except Exception:
        return None


def read_pair(path, input_size, scale):
    target_img = read_image(path, size=(input_size[0] * scale, input_size[1] * scale))
    if target_img is None:
        return None
    input_img = skimage.transform.resize(target_img, input_size)
    return input_img, target_img


class TrainDatasetSequence(Sequence):
    def __init__(self,
                 base_train_path,
                 batch_size=32,
                 input_size=(32, 32),
                 scale=3):
        self.batch_size = batch_size
        self.paths      = shuffle(self._get_image_paths(base_train_path))
        self.input_size = input_size
        self.scale      = scale

    def _get_image_paths(self, base_path):
        image_paths = []
        for dirpath in dir_listing(base_path):
            image_paths += file_listing(dirpath, extension='jpg')
        return image_paths

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        X, y = [], []
        count = 0
        for image_path in cycle(self.paths[idx * self.batch_size:]):
            pair = read_pair(image_path, self.input_size, self.scale)
            if pair is None:
                continue
            count += 1
            X.append(pair[0])
            y.append(pair[1])
            if count >= self.batch_size:
                break
        return np.array(X), np.array(y)


class TestDatasetSequence(Sequence):
    def __init__(self,
                 base_test_path,
                 batch_size=32,
                 input_size=(32, 32),
                 scale=3):
        self.batch_size = batch_size
        self.paths      = file_listing(base_test_path, extension='JPEG')
        self.input_size = input_size
        self.scale      = scale

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        X, y = [], []
        count = 0
        for image_path in cycle(self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]):
            pair = read_pair(image_path, self.input_size, self.scale)
            if pair is None:
                continue
            count += 1
            X.append(pair[0])
            y.append(pair[1])
            if count >= self.batch_size:
                break
        return np.array(X), np.array(y)
