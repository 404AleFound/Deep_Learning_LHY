import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_path):
    steps, losses, accs = [], [], []
    with open(log_path) as f:
        for line in f:
            parts = line.split(';')
            steps.append(int(parts[0].split(':')[1]))
            losses.append(float(parts[1].split(':')[1]))
            accs.append(float(parts[2].split(':')[1]))
    return steps, losses, accs

steps, losses, accs = parse_log_file('./logs/20250810_185705/log_train.txt')


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, 'b-', label='Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(False)

plt.subplot(1, 2, 2)
plt.plot(steps, accs, 'r-', label='Validation Accuracy')
plt.xlabel('Training Steps')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.grid(False)

plt.tight_layout()
plt.show()