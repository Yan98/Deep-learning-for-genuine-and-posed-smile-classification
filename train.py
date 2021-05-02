#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:33:11 2020

@author: Yan
"""

from util import train
import os
import torch
from data import UVANEMODataGenerator,BBCDataGenerator,SPOSDataGenerator,MMIDataGenerator
import numpy as np
from model import DeepSmileNet

#training function for different database

def train_UVANEMO(epoch,lr,label_path,frame_path,frequency,batch_size,sub,file_name  = "uvanemo_training"):
    
    '''
        training function for UVANEMO databases
   
        Parameters
        ----------
        epoch      : int,how many epochs used to train the network
        lr         : float,learning rate for the network
        label_path : str, the path that contains the details of video and label in the cross validation
        frame_path : str, the path that contains processed frames
        frequency  : int, how many frames used per second
        batch_size : int, the number of data for mini-batch size
        file_name  : path, the place to store the trained nn weights.
    '''
    
    for file in os.listdir(label_path): 
        
        current_path = os.path.join(label_path,file)
        if not os.path.isdir(current_path):
            continue
        
        train_labels = os.path.join(current_path,"train.json")
        params = {"label_path": train_labels,
                  "frame_path": frame_path,
                  "frequency" : frequency} 
        dg = UVANEMODataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=batch_size,shuffle=True)
        
        
        test_labels    = os.path.join(current_path,"test.json")
        params = {"label_path"   : test_labels,
                  "frame_path"   : frame_path,
                  "test": True,
                  "frequency": frequency} 
        
        test_generator = torch.utils.data.DataLoader(UVANEMODataGenerator(**params),batch_size=32,shuffle=True)
        
        train(epoch,lr,DeepSmileNet(re = sub),file_name,training_generator,test_generator,file)
        
        
def train_SPOS(epoch,lr,frame_path,frequency,batch_size,sub,file_name  = "spos_training"):
    
    '''
        training function for SPOS databases, 7 cross validation
   
        Parameters
        ----------
        epoch      : int,how many epochs used to train the network
        lr         : float,learning rate for the network
        frame_path : str, the path that contains processed frames, the label can be read from the path,
                    thus no label path required
        frequency  : int, how many frames used per second
        batch_size : int, the number of data for mini-batch size
        file_name  : path, the place to store the trained nn weights.
    '''
    
    for index,name in enumerate(['tomas', 'nelly', 'riku', 'yi', 'ying', 'rui', 'xiaopeng']): 
        c  = 0
        c2 = 0
        traind = dict()
        testd  = dict()
        for i in ["posed","spontaneous"]:
            path = os.path.join(frame_path,i,"happy")
            
            for j in os.listdir(path):
                
                k = os.path.join(path,j)
                if not os.path.isdir(k):
                    continue
                
                if name in k:
                    testd[c2] = k
                    c2+=1
                else:
                    traind[c] = k
                    c+=1
        
        params = {"fold": traind,
                  "frequency" : 5} 
        
        dg = SPOSDataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=batch_size,shuffle=True)
        
        params = {"fold": testd,
                  "test": True,
                  "frequency" : 5} 
        
        test_generator = torch.utils.data.DataLoader(SPOSDataGenerator(**params),batch_size=32,shuffle=True)
        
        train(epoch,lr,DeepSmileNet(re = sub),file_name,training_generator,test_generator,name)
        
def train_BBC(epoch,lr,frame_path,frequency,batch_size,sub,file_name  = "bbc_training"):
    
    '''
        training function for BBC databases, 10 cross validation
   
        Parameters
        ----------
        epoch      : int,how many epochs used to train the network
        lr         : float,learning rate for the network
        frame_path : str, the path that contains processed frames, the label can be read from the path,
                    thus no label path required
        frequency  : int, how many frames used per second
        batch_size : int, the number of data for mini-batch size
        file_name  : path, the place to store the trained nn weights.
    '''
    
    # subjects
    a = list(range(20))
    np.random.seed(12)
    np.random.shuffle(a)
    b = a[10:]
    a = a[:10]
    c = list(zip(a,b))
    from copy import deepcopy
    for name,file2 in enumerate(c): 
        
        file  = deepcopy(c)
        file.remove(file2)
        
        params = {"fold": file,
                  "frame_path": frame_path,
                  "frequency" : frequency} 
        dg = BBCDataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=batch_size,shuffle=True)
        
        params = {"fold": file2,
                  "frame_path": frame_path,
                  "test": True,
                  "frequency" : frequency} 
        
        test_generator = torch.utils.data.DataLoader(BBCDataGenerator(**params),batch_size=32,shuffle=True)
        
        train(epoch,lr,DeepSmileNet(re = sub),file_name,training_generator,test_generator,name)   
        
def train_MMI(epoch,lr,label_path,frame_path,frequency,batch_size,sub,file_name  = "mmi_training"):
    
    '''
        training function for MMI databases, 3 cross validation
   
        Parameters
        ----------
        epoch      : int,how many epochs used to train the network
        lr         : float,learning rate for the network
        label_path : str, the path that contains the details of video and label in the cross validation
        frame_path : str, the path that contains processed frames
        frequency  : int, how many frames used per second
        batch_size : int, the number of data for mini-batch size
        file_name  : path, the place to store the trained nn weights.
    '''
    
    #subjects
    fold = [
    ['54','21','46','2','34'],        
    ['61','35','5','44','3'],        
    ['55','49','45','37'],
    ['60','48','41','42'],
    ['53','32','1','40'],
    ['59','43','47','39',],
    ['56','50'],
    ['57','33','30'],
    ['58','36','29'],     
    ]
    
    label_path = "labels"    
    for name, file in enumerate(fold): 
        
        params = {"fold": set(np.concatenate(fold)).difference(set(file)),
                  "label_path": label_path,
                  "frame_path": frame_path,
                  "frequency" : frequency} #15
        dg = MMIDataGenerator(**params)
        training_generator = torch.utils.data.DataLoader(dg,batch_size=batch_size,shuffle=True)
        
        params = {"fold": file,
                  "label_path": label_path,
                  "frame_path": frame_path,
                  "test": True,
                  "frequency" : frequency} 
        
        test_generator = torch.utils.data.DataLoader(MMIDataGenerator(**params),batch_size=32,shuffle=True)
        
        train(epoch,lr,DeepSmileNet(re = sub),file_name,training_generator,test_generator,name)
        
        
        