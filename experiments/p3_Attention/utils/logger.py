# logger.py -- set the loggers' format
# Le Jiang 
# 2025/8/22

import logging
import re

def create_logger(log_name, log_path, show_time=False):
    # create a logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    # create a handler
    handler = logging.FileHandler(log_path, encoding='utf-8')
    # create a formatter
    if show_time:
        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    else:
        formatter = logging.Formatter(fmt='%(message)s')
    # assemble them
    logger.addHandler(handler)
    handler.setFormatter(formatter)
    return logger

def get_data_train_logger(train_logger_path):
    steps, train_loss, train_acc = [], [], []

    with open(train_logger_path) as file:
        train_logger_lines = file.readlines()
    # step 690 train loss 1.992920 train acc 0.335938
    pattern = r"step (\d+) train loss ([\d.]+) train acc ([\d.]+)"
    for line in train_logger_lines:
        match = re.search(pattern, line)
        if match:
            steps.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_acc.append(float(match.group(3)))

    return steps, train_loss, train_acc

def get_data_val_logger(val_logger_path, n_steps):
    steps, val_loss, val_acc = [], [], []

    with open(val_logger_path) as file:
        val_logger_lines = file.readlines()
    # EPOCH [073|200] VAL LOSS 1.260031 VAL ACC 0.573918
    pattern = r"EPOCH \[(\d+)\|\d+\] VAL LOSS ([\d.]+) VAL ACC ([\d.]+)"
    for line in val_logger_lines:
        match = re.search(pattern, line)
        if match:
            steps.append(int(match.group(1))*n_steps)
            val_loss.append(float(match.group(2)))
            val_acc.append(float(match.group(3)))

    return steps, val_loss, val_acc