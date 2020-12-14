import torch
import torch.nn as nn
from torch import nn, optim, Tensor
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from typing import List, Tuple, Any, Optional, Dict

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, entity_embeddings, num_relations):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.entity_embeddings = entity_embeddings
        self.num_relations = num_relations
        # self.relation_embeddings = relation_embeddings
        
        self.embed_size = self.entity_embeddings.weight.shape[1]
        
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)
        
        self.linear = nn.Linear(self.hidden_size, self.num_relations)

    def forward(self,
                entities: Tensor,
                encoder_output: Tensor):
        
        ### Create embeddings
        embed = self.entity_embeddings(entities)
        embed = embed.view(1, -1, self.embed_size)

        ### Pass through LSTM decoder - NOT SURE ABOUT THIS ARC
        _, (out, _) = self.lstm(embed, encoder_output)
        num_batches, _, hidden_size = out.shape
        out = out.reshape(num_batches, hidden_size)
        out = self.linear(out)
        
        return out