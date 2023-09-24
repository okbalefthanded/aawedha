from pynvml import *
from tensorflow.keras import backend as K
from aawedha.analysis.utils import isfloat
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import zipfile
import tarfile
import random
import torch
import os


def get_device(config=None):
    """Returns compute settings.engine : GPU / TPU

    Returns
    -------
    str
        computer settings.engine for training
    """
    # test if env got GPU
    device = 'GPU'  # default
    if config:
        if 'device' in config:
            return config['device']
    
    devices = [dev.device_type for dev in tf.config.get_visible_devices()]
    if 'GPU' not in devices:
        device = 'CPU'
    if device == 'GPU':
        device = 'cuda'
    return device

def get_gpu_name():
    """Returns the device (GPU) name
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    name = nvmlDeviceGetName(handle)
    nvmlShutdown()
    return name

def get_tpu_address():
    """Return TPU Address

    Returns
    -------
        TPU address
    """
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        print('Found TPU at: {}'.format(TPU_ADDRESS))
    except KeyError:
        print('TPU not found')
    return TPU_ADDRESS

def init_TPU():
    """initialize TPU for training on TPUs

    Returns
    -------
    TPUStrategy
    """
    TPU_ADDRESS = get_tpu_address()
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
    tf.config.experimental_connect_to_cluster(resolver)

    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    # print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)
    return strategy

def log_to_csv(filepath, folder=''):
    """Convert log file into a csv file of results only

    Parameters
    ----------
    filepath : str
        log file file path
    folder : str
        where to store csv file
    """
    accs = []
    subjects = 0

    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            if cnt == 0:
                header = line
            if 'ACC' in line:
                if 'epoch' in line:
                    acc_line = line.split('ACC: ')[-1].split(' ')
                    numbers = [ch.split('[')[-1].split(']')[0] for ch in acc_line]
                    numbers = [float(ch) for ch in numbers if isfloat(ch)]
                    numbers = numbers[:-1]
                    # numbers = float(acc_line[0].split('[')[-1])
                else:
                    acc_line = line.split('ACC: ')[-1]
                    numbers_str = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in acc_line)
                    numbers = [float(i)*100 for i in numbers_str.split()]
                accs.append(numbers)
                subjects += 1
    if isinstance(accs[0], list):
        nfolds = len(accs[0])
    else:
        nfolds = 1
    accs = np.array(accs)
    parts = filepath.split("/")[-1].split('_')
    evl = parts[0]
    dataset = parts[1]
    date = '_'.join(parts[2:-1])
    model_name = header.split('Model')[1].split(' ')[1] 
    metric = 'Accuracy'
    rows = [f'S{s+1}' for s in range(subjects)] 
    columns = [f'Fold {fld+1}' for fld in range(nfolds)]
    df = pd.DataFrame(data=accs, index=rows, columns=columns)
    df.index.name = f"{model_name} / {metric}"
    if folder:
        fname = f"{folder}/{evl}_{dataset}_{metric}_{date}.csv"
    else:
        fname = f"{evl}_{dataset}_{metric}_{date}.csv"
    df.to_csv(fname, encoding='utf-8')
    print(f"csv file saved to : {fname}")

def time_now():
    """Return current time in Year_Month_day_Hour_Minutes_Seconds format.
    """
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def unzip_files(zip_files, store_path):
    """Unzip compressed files and delete zipped files.

    Parameters
    ----------
    zip_files : list of str
        path to zipped files to extract
    store_path : str
        folder where to extract zipped files.
    """
    for zipf in zip_files:
        unzip_single_file(store_path, zipf)
        # zip_ref = zipfile.ZipFile(zipf) # create zipfile object
        # zip_ref.extractall(store_path)  # extract file to dir
        # zip_ref.close() # close file
        os.remove(zipf) # delete zipped file

def unzip_single_file(path, compressed_file):
    save_dir = Path(path)  
    zip_file = zipfile.ZipFile(compressed_file, 'r')
    for files in zip_file.namelist():
        data = zip_file.read(files)
        file_path = save_dir / Path(files).name
        file_path.write_bytes(data)
    zip_file.close()

def untar_files(tar_files, store_path):
    """Untar compressed files and delete tar files.

    Parameters
    ----------
    tar_files : list of str
        path to zipped files to extract
    store_path : str
        folder where to extract zipped files.
    """
    for tfile in tar_files:
        tar = tarfile.open(tfile)
        tar.extractall(store_path)
        tar.close()
        os.remove(tfile)

def set_seed(seed):
    """Reseed random numbers generators with a common seed.

    Parameters
    ----------
    seed : int
       random numbers generators seed
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)

def make_folders(root="aawedha"):
    """Create additional folders where logs, trained models, checkpoints
    and debug logs will be saved.

    Parameters
    ----------
    root : str, optional
        parent folder of the newly created folders, by default "aawedha"
    """
    folder_names = ['checkpoint', 'debug', 'logs', 'results', 'trained']
    for name in folder_names:
        folder = os.path.join(root, name)
        if ':' in folder:
            folder = folder.replace(':', '_')
        if not os.path.isdir(folder):
            os.mkdir(folder)

def cwd():
    """Get current working directory

    Returns
    -------
    str
        current working directory
    """
    cwdir = os.getcwd().split('/')[-1]
    return os.getcwd() if cwdir != 'aawedha' else 'aawedha'

def set_channels_order(order='first'):
    """Set the order of channels in tensors for Keras to
    First or Last.

    Parameters
    ----------
    order : str, optional
        dimension of channels, by default 'first'
    """
    assert order in ('first', 'last')
    K.set_image_data_format(f'channels_{order}')

