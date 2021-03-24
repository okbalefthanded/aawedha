from pynvml import *
from aawedha.analysis.utils import isfloat
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import datetime
import os


def log(fname='logger.log', logger_name='eval_log'):
    """define a logger instance

    Parameters
    ----------
    fname : str
        logger file path
    logger_name : str
        logger name

    Returns
    -------
    logger
        logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    # Create handlers
    f_handler = logging.FileHandler(fname, mode='a')
    f_handler.setLevel(logging.DEBUG)
    # Create formatters and add it to handlers
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    # Add handlers to the logger
    
    if len(logger.handlers) > 0:
        for hdl in logger.handlers:
            logger.removeHandler(hdl)

    logger.addHandler(f_handler)
    # c_handler = logging.StreamHandler()
    # c_handler.setLevel(logging.WARNING)
    # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    # c_handler.setFormatter(c_format)
    # logger.addHandler(c_handler)
    return logger


def get_gpu_name():
    """Returns the device (GPU) name
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    name = nvmlDeviceGetName(handle).decode('UTF-8')
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
    """[summary]
    """
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')