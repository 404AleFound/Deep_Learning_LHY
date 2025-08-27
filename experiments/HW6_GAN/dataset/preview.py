# preview.py -- preview the data of faces
# Le Jiang
# 2025/8/10

from matplotlib import pyplot as plt

img = plt.imread('./data/faces/1.jpg')
plt.imshow(img)
plt.show()