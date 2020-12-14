import time

import torch
import torch.nn as nn
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from typing import List, Tuple, Any, Optional, Dict

from training.data import Instance, Data, Dictionary, triples_to_indices
from training.dataset import CLUTRRdata
from models.encoders import EncoderLSTM
from models.decoders import DecoderLSTM


def train(encoder, decoder, trainloader, epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model...')
    
    loss_fn = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        step = 1
        for i, data in enumerate(trainloader):
            story = data['story']
            query = data['query']
            target = data['target'].view(-1)
        
            encoder_output = encoder(story, None)
            output = decoder(query, encoder_output)
            
            loss = loss_fn(output, target)
            
            # Double check that this is needed
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            loss.backward()
            
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            epoch_loss += loss.item()
            
            if step%100 == 0:
                print(f'Running epoch [{epoch+1}/{epochs}],'
                    f'Step: [{step}/{len(trainloader)}],'
                    f'Loss: {epoch_loss:.2f}               ', end="\r", flush=True)
            step += 1
        
        print(f'Epoch [{epoch+1}/{epochs}], Train loss: {epoch_loss:.2f}                        ')
    
    print("Finished training\n")
    return

def predict_raw(instance, encoder, decoder, dictionary):
    '''
    Predict the relation from the query given a model
    Takes a single data Instance (raw, ie not yet transformed in indices)
    '''
    print(f'Story: {instance.story}')
    story, _, _ = triples_to_indices(dictionary, instance.story)
    _, query, _ = triples_to_indices(dictionary, [instance.target])
    
    # Add dimension to fit with encoder/decoder input
    story.unsqueeze_(0)
    query.unsqueeze_(0)
    
    encoder_output = encoder(story, None)
    output = decoder(query, encoder_output)
    
    pred = dictionary.idx2relation[int(torch.argmax(output))]
    
    print(f'Target: {instance.target}')
    print(f'Predicted relation: {pred}')
    return