import time

import torch
import torch.nn as nn
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

import numpy as np
from typing import List, Tuple, Any, Optional, Dict

from data.raw_data import Instance, Data
from data.data_utils import Dictionary, triples_to_indices, collate
from models.encoders import EncoderLSTM
from models.decoders import DecoderLSTM


def train(encoder, decoder, train_data, test_data, epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model...')
    
    loss_fn = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    batch_size = 20
    trainloader = DataLoader(dataset = train_data,
                        batch_size = batch_size,
                        shuffle = True,
                        collate_fn=collate)
    testloader = DataLoader(dataset = test_data,
                        batch_size = batch_size,
                        shuffle = True,
                        collate_fn=collate)
    
    # Evaluating initial loss/accuracy
    train_loss, train_acc = evaluate(encoder, decoder, trainloader, loss_fn)
    val_loss, val_acc = evaluate(encoder, decoder, testloader, loss_fn)
    print(f'Initial values: \n' +
          f'Train loss: {train_loss*1000:.2f} | Train acc: {train_acc:.2f}% ' +
          f'| Test loss: {val_loss*1000:.2f} | Test acc: {val_acc:.2f}%')
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        epoch_loss = 0.0
        
        step = 1
        
        correct_predictions = 0
        total_samples = 0
        for i, data in enumerate(trainloader):
            story = data['story'].to(device)
            query = data['query'].to(device)
            target = data['target'].view(-1).to(device)
        
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
            
            # For accuracy
            _, pred = torch.max(output, 1)
            total_samples += target.shape[0]
            correct_predictions += (pred == target).sum().item()
            
            if step%100 == 0:
                print(f'Running epoch [{epoch+1}/{epochs}],'
                    f'Step: [{step}/{len(trainloader)}],'
                    f'Loss: {epoch_loss:.2f}               ', end="\r", flush=True)
            step += 1
        
        epoch_loss /= len(train_data)
        
        val_loss, val_acc = evaluate(encoder, decoder, testloader, loss_fn)
        print(f'Epoch [{epoch+1}/{epochs}],'+
              f'Train loss: {epoch_loss*1000:.4f},' +
              f'Test loss: {val_loss*1000:.4f} ' +
              f'Train acc: {correct_predictions/total_samples*100:.2f}% ' +
              f'Test acc: {val_acc:.2f}%')
    
    print("Finished training\n")
    return

def predict_raw(instance, encoder, decoder, dictionary):
    '''
    Predict the relation from the query given a model
    Takes a single data Instance (raw, ie not yet transformed in indices)
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Story: {instance.story}')
    story, _, _ = triples_to_indices(dictionary, instance.story)
    _, query, _ = triples_to_indices(dictionary, [instance.target])
    
    # Add dimension to fit with encoder/decoder input
    story.unsqueeze_(0)
    query.unsqueeze_(0)
    
    story = story.to(device)
    query = query.to(device)
    
    encoder_output = encoder(story, None)
    output = decoder(query, encoder_output)
    
    pred = dictionary.idx2relation[int(torch.argmax(output))]
    
    print(f'Target: {instance.target}')
    print(f'Predicted relation: {pred}')
    return

def evaluate(encoder, decoder, testloader, loss_function):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    decoder.eval()
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        loss = 0.0
        for i, data in enumerate(testloader):
            story = data['story'].to(device)
            query = data['query'].to(device)
            target = data['target'].view(-1).to(device)
            
            encoder_output = encoder(story, None)
            output = decoder(query, encoder_output)
            
            loss += loss_function(output, target)
            
            # Accuracy
            _, pred = torch.max(output, 1)
            total_samples += target.shape[0]
            correct_predictions += (pred == target).sum().item()
    loss /= total_samples
    accuracy = correct_predictions/total_samples*100
    return loss, accuracy