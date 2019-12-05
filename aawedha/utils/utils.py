import logging


def log(fname='', logger_name=''):
    '''
    '''
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
