import time

import torch
import torch.nn as nn
from torch import nn, optim, Tensor
import torch.nn.functional as F

import numpy as np
from typing import List, Tuple, Any, Optional, Dict

from data.raw_data import Instance, Data
    
class Dictionary:
    '''
    A dictionary to convert entities/predicates to indices and vice versa
    
    Argument:
        triples: triples from which to extract all entities/predicates
                used to build the dictionary
    '''
    def __init__(self,
                 data):
        
        # Entities dic
        self.entity2idx = {entity: idx for idx, entity in enumerate(data.entity_lst)}
        self.idx2entity = {idx: entity for entity, idx in self.entity2idx.items()}
        self.num_entities = len(self.entity2idx)
        
        # Predicate dic
        self.relation2idx = {relation: idx for idx, relation in enumerate(data.relation_lst)}
        self.idx2relation = {idx: relation for relation, idx in self.relation2idx.items()}
        self.num_relations = len(self.relation2idx)
        
# Need to figure out whether to return one tensor or split it in two
def triples_to_indices(dictionary, triples):
    
    indices = np.array([[dictionary.entity2idx[s], dictionary.entity2idx[o], dictionary.relation2idx[r]] for s, r, o in triples])
    indices = torch.tensor(indices, dtype=torch.long)
    
    entities_indices = indices[:,:2]
    relation_indices = indices[:,2]

    return indices, entities_indices, relation_indices