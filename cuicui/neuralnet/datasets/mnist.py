import os
import numpy as np

from .utils import get_cache_dir, download_url


def load_data(normalize=True, one_hot_label=False):
    """
    """
    cache_dir = get_cache_dir()
    path = os.path.join(cache_dir, 'mnist.npz')

    if not os.path.exists(path):
        download_mnist(path)

    with np.load(path, allow_pickle=True) as f:
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']

    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    if one_hot_label:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


def download_mnist(path):
    """
    """
    url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
    hash_value = '731c5f0a9d14c4d76d19b0e6f314ee0e'  # MD5 file hash
    download_url(url, path, hash_value=hash_value, hash_type='md5')


def to_categorical(y, num_classes):
    """
    """
    return np.eye(num_classes)[y]
