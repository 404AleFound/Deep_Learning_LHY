# logger.py --
# Le Jiang
# 2025/8/25

import logging

def create_filelogger(logger_name, logger_path):
    # define logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # define hander
    hander = logging.FileHandler(f'{logger_path}/{logger_name}.txt')
    # define formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # assemble
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    return logger