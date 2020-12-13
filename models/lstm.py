import torch
import torch.nn as nn

class LSTM(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(LSTM, self).__init__()
        
        # Create embeddings
        # entity_embeddings = nn.Embedding(data.nb_entities, rank, sparse=True)
        # predicate_embeddings = nn.Embedding(data.nb_predicates, rank, sparse=True)

    def forward(self, x):

        return x