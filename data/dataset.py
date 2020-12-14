import time

import torch
import torch.nn as nn

from typing import List, Tuple, Any, Optional, Dict

from data.raw_data import Instance, Data
from data.data_utils import Dictionary, triples_to_indices

class CLUTRRdata(torch.utils.data.Dataset):
    '''
    Dataset with CLUTRR data in indices.
    Includes:
    story: (len_story, 3) tensor
    query: (1, 2) tensor
    target: (1) tensor
    '''
    def __init__(self,
                 data: List[Instance],
                 dictionary: Dictionary):
        super(CLUTRRdata, self).__init__()
        self.data = data
        self.dictionary = dictionary
        self.instances = []
        
        for idx, instance in enumerate(data):
            instance_in_indeces = {}
            
            story, _, _ = triples_to_indices(dictionary, instance.story)
            _, query, target = triples_to_indices(dictionary, [instance.target])
            self.instances += [{'story': story,
                                'query': query,
                                'target': target}]
        

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)