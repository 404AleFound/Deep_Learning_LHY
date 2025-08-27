import numpy as np
import matplotlib.pyplot as plt
test_data_path = './data/ml2022spring-hw8/testingset.npy'
train_data_path = './data/ml2022spring-hw8/trainingset.npy'

test_data = np.load(test_data_path)
print(test_data.shape)# (19636, 64, 64, 3)

plt.imshow(test_data[0])
plt.show()