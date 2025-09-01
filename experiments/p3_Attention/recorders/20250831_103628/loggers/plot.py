# plot.py -- plots the loss curve of training and validation, plots the acc curve of training and validation 
# Le Jiang
# 2025/8/28

from matplotlib import pyplot as plt
import os
import sys
file_ab_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_ab_path, '..', '..','..')))
from utils import get_data_train_logger, get_data_val_logger

timestamp = '20250831_103628'
steps_train, loss_train, acc_train = get_data_train_logger(f'./recorders/{timestamp}/loggers/train.log')
steps_val, loss_val, acc_val = get_data_val_logger(f'./recorders/{timestamp}/loggers/eval.log', n_steps=22)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Over Steps')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='y', labelcolor='b')

ax1.plot(steps_train, loss_train, 'b-', linewidth=1, label='loss_train')
ax1.plot(steps_val, loss_val, 'r-', linewidth=1, label='loss_val')

ax2.set_xlabel('Steps')
ax2.set_ylabel('Acc')
ax2.set_title('Acc Over Steps')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.tick_params(axis='y', labelcolor='b')

ax2.plot(steps_train, acc_train, 'b-', linewidth=1, label='acc_train')
ax2.plot(steps_val, acc_val, 'r-', linewidth=1, label='acc_val')

plt.legend()
plt.tight_layout()
plt.show()