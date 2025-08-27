# preview.py -- have a quick look at data/Voxceleb2_Part

# datadir structure:
# Voxceleb2_Part
#   |__ metadata.json    
#   |__ testdata.json     
#   |__ mapping.json     
#   |__ uttr-{random string}.pt   

# metadata.json
#   "n_mels": 40， mel图谱的维度.
#   "speakers": 字典. 
#   Key: speaker ids.
#   value: "feature_path"-特征文件 and "mel_len"-特征的长度

import json
import torch

metadata_path = './data/Voxceleb2_Part/metadata.json'
with open(metadata_path) as file:
    metadata = json.load(file)

keys = metadata.keys()
print('================================================================')
print('the keys in metadata:', keys)
print(('================================================================'))
print("metadata['n_mels']:\n", 
      'type: ', type(metadata['n_mels']),'\n',
      'value: ', metadata['n_mels'])
print(('================================================================'))
print("metadata['speakers']:\n",
      'type: ', type(metadata['speakers']),'\n',
      'length:', len(metadata['speakers']), '\n',
      'the first 5 elements:', list(metadata['speakers'].keys())[:5]
      ) 
print(('================================================================'))
print("metadata['speakers']['id03074']:",'\n',
      'type: ', type(metadata['speakers']['id03074']),'\n',
      'length: ', len(metadata['speakers']['id03074']),'\n'
      'the first 5 elements: ')
for i in range(5):
    print(metadata['speakers']['id03074'][i])
print(('================================================================'))
datadir = './data/Voxceleb2_Part/'
speaker_id03074 = torch.load(datadir + metadata['speakers']['id03074'][0]['feature_path'])
print("metadata['speakers']['id03074'][0]['feature_path']",'\n',
      'type: ', type(speaker_id03074), '\n',
      'shape: ', speaker_id03074.shape, '\n',
      'value: \n', speaker_id03074)

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