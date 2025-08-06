# dataset.py -- pack the data into dataset

import torch
from torch.utils.data import Dataset
# speaker[1]->id(label)
#     feature[0]->tensor
#           tensor.shape->torch.size(m,40)
#     ...
#     feature[i]->tensor
#           tensor.shape->torch.size(m,40)
#     ...
# ...
# speaker[i]->id(label)
#     feature[0]->tensor
#           tensor.shape->torch.size(m,40)
#     ...
#     feature[i]->tensor
#           tensor.shape->torch.size(m,40)
#     ...
# speaker[600]->id(label)
#     feature[0]->tensor
#           tensor.shape->torch.size(m,40)
#     ...
#     feature[i]->tensor
#           tensor.shape->torch.size(m,40)
#     ...

class Voice(Dataset):
    def __init__(self):
        super(Voice).__init__()
