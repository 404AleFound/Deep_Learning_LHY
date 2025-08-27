# logger.py -- set the loggers' format
# Le Jiang 
# 2025/8/22

import logging

def create_logger(log_name, log_path, show_time=False):
    # create a logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    # create a handler
    handler = logging.FileHandler(log_path)
    # create a formatter
    if show_time:
        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    else:
        formatter = logging.Formatter(fmt='%(message)s')
    # assemble them
    logger.addHandler(handler)
    handler.setFormatter(formatter)
    return logger