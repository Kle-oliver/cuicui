import os
import hashlib
import requests
from tqdm.auto import tqdm


def get_cache_dir():
    """
    """
    cache_dir = os.path.expanduser('~/cuicui/neuralnet/dataset')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_url(url, dest_path, hash_value=None, hash_type='md5'):
    """
    """
    if os.path.exists(dest_path):
        if hash_value and check_hash(dest_path, hash_value, hash_type):
            return
        else:
            os.remove(dest_path)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    desc = f'Downloading... {os.path.basename(dest_path)}'
    with open(dest_path, 'wb') as file, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=desc) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

    if hash_value and not check_hash(dest_path, hash_value, hash_type):
        raise ValueError('Falha na verificação de integridade do arquivo.')


def check_hash(file_path, hash_value, hash_type='md5'):
    """
    """
    hash_func = getattr(hashlib, hash_type)()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest() == hash_value


def extract_archive(file_path, extract_to=None):
    """
    """
    import tarfile
    import zipfile
    if extract_to is None:
        extract_to = os.path.dirname(file_path)
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f'File format not supported: {file_path}')
