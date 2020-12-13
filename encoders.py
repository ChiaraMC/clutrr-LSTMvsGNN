import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from typing import List, Tuple, Any, Optional, Dict

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, entity_embeddings, relation_embeddings):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        
        self.embed_size = self.entity_embeddings.weight.shape[1]
        assert self.embed_size == self.relation_embeddings.weight.shape[1]
        
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)
        self.encoder = PytorchSeq2VecWrapper(self.lstm)

    def forward(self,
                story: List[],
                hidden):
        
        
        
        
        
        output = embedded
        output, hidden = self.gru(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)