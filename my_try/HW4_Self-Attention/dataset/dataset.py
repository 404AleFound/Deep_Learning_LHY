# dataset.py -- pack the data into dataset

import torch
from torch.utils.data import Dataset
import json
import doctest
import os
import random

class Voice(Dataset):
    '''
    ---
    TEST
    ---
    >>> voice = Voice('./data/Voxceleb2_Part', is_seg=False)
    >>> voice[0][0].shape
    torch.Size([435, 40])
    >>> voice[0][1]
    436
    >>> voice[4][1] == voice[7][1]
    True
    '''
    def __init__(self, datadir, is_seg=True, seg_len=400):
        super(Voice,self).__init__()
        self.is_seg = is_seg
        metadata_file = os.path.join(datadir, 'metadata.json')
        mapping_file = os.path.join(datadir, 'mapping.json')
        with open(metadata_file) as file:
            self.dict_metadata = json.load(file)
        with open(mapping_file) as file:
            self.dict_mapping = json.load(file)
        self.dict_speakers = self.dict_metadata['speakers']
        self.n_mels = self.dict_metadata['n_mels']
        self.seg_len = seg_len
        # contain two elements: feature and id
        self.voice = []
        for key, seqs_info in self.dict_speakers.items():# seqs_info is a list
            for seq_info in seqs_info:# seq_info is a dict
                seq_file = os.path.join(datadir, seq_info['feature_path'])
                seq = torch.load(seq_file)
                label = self.dict_mapping['speaker2id'][key]
                len_seq = seq_info['mel_len']
                self.voice.append([seq, label, len_seq])

    def __getitem__(self, index):
        seq, label, len_seq = self.voice[index]
        if len_seq > self.seg_len and self.is_seg:
            begin = random.randint(0, len_seq - self.seg_len)
            seq = seq[begin:(begin+400)]
        return seq, label
    
    def __len__(self):
        return len(self.voice)
        

if __name__ == '__main__':
    doct = doctest.testmod(verbose=True)