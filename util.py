#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:14:05 2020

@author: Yan
"""

from PIL import Image
import torch
import torch.nn as nn
import io


##############################
######## help functions ######
##############################

def out_put(string,verbose):
    '''
    Help function for verbose,
    output the string to destination path
   
    Parameters
    ----------
    string  :str,  the string to output
    verbose :str, the path to store the output
    '''
    with open(f"verbose.txt","a") as f:
        f.write(string + "\n")


def read_image(zipfile,name):
    '''
    read the image from zipfile
   
    Parameters
    ----------
    zipfile  :ZipFile, the zipfile
    name     :str, the path to read the image
    '''
    return Image.open(io.BytesIO(zipfile.read(name)))


def weighted_binary_cross_entropy(output, target, weights=None):
    '''
    Implementation of weighted binary cross entropy
   
    Parameters
    ----------
    output   :torch.tensor, predict probability
    target   :torch.tensor, the acctual lables
    weights  :tuple, the weight for each classes
    '''
        
    loss = weights[0] * ((1 - target) * torch.log(1 - output)) + weights[1] * (target * torch.log(output)) 
    
    return torch.neg(torch.mean(loss))
    
def train(epochs,lr,net,file_name,training_generator,test_generator,file,weights = None):
    
    '''
    train the networks and validate the model on different data, will use cuda if there has one
   
    Parameters
    ----------
    epochs              :int, the number of epochs for training.
    net                 :torch.module, the networks need to be train
    file_name           :str, the path to save the model weights and the path to output training informations
    training_generator  : torch data generaters, used for generate minibatch when training
    test_generators     : torch data generaters, used for validation when training
    file                : str,the name of current validation generaters
    weights             : tuple,the weights used for binary cross entropy
    '''
    
    #used to record the accuracy of each epoch
    con = []    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        out_put(f"use {torch.cuda.device_count()} GPUS",file_name)
        net = nn.DataParallel(net)
    elif torch.cuda.device_count() == 1:
        out_put(f"use one gpu",file_name)
    
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = (nn.BCELoss() if weights == None else weighted_binary_cross_entropy)
    #used to record the best accuracy
    best_accuracy = 0
    
    for epoch in range(epochs):
        train_loss = 0 
        pred_label = []
        true_label = []
        
        for x, y,s in training_generator:
            index = s.min().item()
            s = s - s.min()
            x =  x.type(torch.FloatTensor)[:,index:]
            y =  y.type(torch.FloatTensor)
            
            if torch.cuda.device_count() > 0:
                x  = x.to(device)
                y  = y.to(device)
                s = s.to(device)
            pred = net(x,s)
            
            pred_y = (pred >= 0.5).float().to(device).data
            pred_label.append(pred_y)
            true_label.append(y)
            
            if weights == None:
                loss = loss_func(pred, y)
            else:
                loss = loss_func(pred, y,weights = weights)
            
            for W in net.parameters():
                loss += 0.001 * W.norm(2)
            
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy: ' + str(train_accuracy.item()),file_name)
               
        net.eval()
        
        pred_label = []
        true_label = []
        
        for x, y, s in test_generator:
            index = s.min().item()
            x =  x.type(torch.FloatTensor)[:,index:]
            y = y.type(torch.FloatTensor)
            if torch.cuda.device_count() > 0:
                x = x.to(device)
                y = y.to(device)
                s = s.to(device)
            pred_y = (net(x,s) >= 0.5).float().to(device).data
            pred_label.append(pred_y)
            true_label.append(y)
        
        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)
        
        test_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        con.append([epoch,test_accuracy])
        out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(train_loss) + '| test accuracy: ' + str(test_accuracy.item()),file_name)
        
        if test_accuracy > best_accuracy:
            filepath = f"{file_name}/{file}-{epoch:}-{loss}-{test_accuracy}.pt"
            torch.save(net.state_dict(), filepath)
            best_accuracy = test_accuracy  
            
        net.train()
        
    best_v = max(con,key = lambda x:x[1])
    
    perf = f"best accuracy  for {file} is {best_v[1]} in epoch {best_v[0]}" + "\n"
    
    out_put(perf,file_name)