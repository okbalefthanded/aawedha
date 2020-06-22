from pynvml import *
import logging


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
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    # Add handlers to the logger
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