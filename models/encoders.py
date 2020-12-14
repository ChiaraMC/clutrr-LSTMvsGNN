import torch
import torch.nn as nn
from torch import nn, optim, Tensor
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from typing import List, Tuple, Any, Optional, Dict

class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size, entity_embeddings, relation_embeddings):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.entity_embeddings.requires_grad = False
        self.relation_embeddings.requires_grad = False
        
        self.embed_size = self.entity_embeddings.weight.shape[1]
        assert self.embed_size == self.relation_embeddings.weight.shape[1]
        
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = 1,
                            batch_first = True,
                            bidirectional = False)
        
        self.encoder = PytorchSeq2VecWrapper(self.lstm) #Any benefit to using this
        # vs. doing it manually? Seems to take roughly same time

    def forward(self,
                story: Tensor,
                hidden):
        
        ### Create embeddings
        # Separete entities and relations, embed them, and put them back together
        batches, story_len, _ = story.shape
        entities = story[:,:,:2]
        relations = story[:,:,2].view(batches, 1, -1)
        
        # Embed them
        ent_embed = self.entity_embeddings(entities)
        rel_embed = self.relation_embeddings(relations)
        
        # Put them back together and flatten in a (1, embed_size) tensor
        embed = torch.cat((ent_embed, rel_embed), dim=1)
        embed = embed.view(batches, -1, self.embed_size)

        ### Pass through LSTM encoder
        # Manual way 
        _, out = self.lstm(embed, None)
        # out = out[0][:,-1,:] # Get only last hidden layer
        
        # Using allennlp    
        #out = self.encoder(embed, None)
        
        return out
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)