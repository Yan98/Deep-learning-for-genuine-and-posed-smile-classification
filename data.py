#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:58:33 2020

@author: Yan
"""

import torch
import os
import json
import numpy as np
from PIL import Image
import zipfile
from util import read_image


#################################################
######## dataloader for different database ######
#################################################

def zipToName(data):
    
    '''
    read the data, and map the name of videos to each frame
   
    Parameters
    ----------
    
    data : zipfile, the file that contains all frames
    
    '''
    
    data_name = dict()
    
    for file in data.namelist():
        if len(file.split("/")) != 2:
            continue
        name,f = file.split("/")
        if name in data_name:
            data_name[name].append(f)
        else:
            data_name[name] = [f]   
            
    return data_name,max([len(i) for i in data_name.values()])

def loadframes(data_name,data,name,frequency,max_len):
    
    '''
    load and sample the frames according to frequency, resize the frames
   
    Parameters
    ----------
    data           : zipfile, the loaded zip of input frames
    name           : str, the name of video to load 
    data_name      : dict, match the name of each videos to its frames
    max_len        : int, the maximum length of sampled videos
    frequency      : the frequency used to sample the frames, which decide how many frames used to represent 
                     the videos
    
    '''
    
    values = []
    files = [f for f in data_name[name] if f.endswith(".jpg") and f.replace(".jpg","").isnumeric()]
    files = sorted(files, key = lambda x : int(x.replace(".jpg","")))
    last = [files[-1]] if len(files) % frequency >= 2/3 * frequency else []
    start = 0
    files = np.concatenate([np.array(files)[start::frequency],last])
    
    l  = 0
    
    pad_x = np.zeros((max_len,3,48,48)) 
    for frame in files:
        
        try:
            
            cur  = read_image(data, os.path.join(name,frame))
        except:
            cur  = read_image(data, name+"/"+frame)
        cur = np.swapaxes(np.asarray(cur.resize((48,48))), 2, 0)
        values.append(cur)
        l+=1
    
    x = np.array(values)
    l = max_len - l
    pad_x[l:] = x
    
    return pad_x,l

class UVANEMODataGenerator(torch.utils.data.Dataset):
    
    '''
    The minibatch generater for UVA-NEMO database used when training the models
    
    Parameters
    ----------
    
    frame_path : str, path, the path that contains the zipped frames.
    label_path : str, path, the label for each videos
    frequency  : the frequency used to sample the frames, which decide how many frames used to represent 
                 the videos
    test       : boolean, default false, whether it is the generator used for test
    scale      : the ratio used to scale the value of each frames
    
        
    Attributes
    ----------
    data           : zipfile, the loaded zip of input frames
    data_name      : dict, match the name of each videos to its frames
    number_of_data : int, the number of videos used
    max_len        : int, the maximum length of sampled videos
    index_name_dic : dict, map ints to the video name
    '''
    
    def __init__(self,frame_path,label_path, frequency = 5,test = False ,scale = 255):
        
        self.data      = zipfile.ZipFile(frame_path)
        self.data_name = dict()
        self.frequency         = frequency
        self.label_path        = label_path
        self.scale             = scale
        self.test              = test
        self.__dataset_information()
        
    def __dataset_information(self):
        
        '''
        Count how many videos are in the folder, map video names to each frames,
        map ints to video names
   
        Parameters
        ----------
        '''
        
        self.data_name,self.max_len = zipToName(self.data)
        self.max_len = self.max_len//self.frequency + 2
        self.numbers_of_data    = 0
        self.index_name_dic = dict()        
        
        with open(self.label_path) as f:
            labels = json.load(f)
        for index,(k,v) in enumerate(labels.items()):
            self.index_name_dic[index] = [k,v]
        self.numbers_of_data = index + 1
                     
                  
    def __len__(self):
        
        '''
        return the length of videos
   
        Parameters
        ----------
        '''
        
        return self.numbers_of_data
    
    def __getitem__(self,idx):
        '''
        Given the int, load the correpsonding frames and labels
   
        Parameters
        ----------
        '''
        
        ids    =  self.index_name_dic[idx]
        name,label = ids
        y = np.zeros(1)
        y[0] = label
        pad_x,l = loadframes(self.data_name, self.data, name, self.frequency, self.max_len)
        
        return pad_x/self.scale,y,l
    
    
    
class BBCDataGenerator(torch.utils.data.Dataset):
    
    '''
    The minibatch generater for BBC database used when training the models
    
    Parameters
    ----------
    
    frame_path : str, path, the path that contains the zipped frames.
    fold       : list, contains the index of videos that used for training
    frequency  : the frequency used to sample the frames, which decide how many frames used to represent 
                 the videos
    test       : boolean, default false, whether it is the generator used for test
    scale      : the ratio used to scale the value of each frames
    
        
    Attributes
    ----------
    data           : zipfile, the loaded zip of input frames
    data_name      : dict, match the name of each videos to its frames
    number_of_data : int, the number of videos used
    max_len        : int, the maximum length of sampled videos
    index_name_dic : dict, map ints to the video name
    '''
    
    def __init__(self,frame_path, fold = None, frequency = 5,test = False ,scale = 255):
        
        self.fold     = np.array(fold).flatten()
        
        self.data     = zipfile.ZipFile(frame_path)
        
        self.data_name,self.max_len = zipToName(self.data)
        self.max_len           = self.max_len//frequency + 2
        self.file_dict         = dict()
        self.frequency         = frequency
        self.numbers_of_data   = len(self.fold)
        self.scale             = scale
        self.test              = test
    
    def __len__(self):
        
        return self.numbers_of_data
    
    def __getitem__(self,idx):
        ids =  self.fold[idx]
        y = np.zeros(1)
        name = sorted(list(self.optical_file_name.keys()))[ids]
        if "Genuinesmilecontent" in name:
            y[0] = 1
        pad_x,l = loadframes(self.data_name, self.data, name, self.frequency, self.max_len)
        
        return pad_x/self.scale,y,l
     

class MMIDataGenerator(torch.utils.data.Dataset):
    
    '''
    The minibatch generater for MMI database used when training the models
    
    Parameters
    ----------
    
    fold       : list, contains the subjects used for this generator
    data_path  : str, path, the path that contains the zipped frames.
    label_path : str, path, the label for each videos
    frequency  : tuple,[int,int] the frequency used to sample the frames, which decide how many frames used to represent 
                 the videos. In the MMI databases, different fps are used. 
    test       : boolean, default false, whether it is the generator used for test
    scale      : the ratio used to scale the value of each frames
    
        
    Attributes
    ----------
    data           : zipfile, the loaded zip of input frames
    data_name      : dict, match the name of each videos to its frames
    number_of_data : int, the number of videos used
    max_len        : int, the maximum length of sampled videos
    index_name_dic : dict, map ints to the video name
    '''
    
    def __init__(self,fold, data_path, label_path, frequency = (5,6),test = False ,scale = 255):
        
        self.fold   = fold
        
        self.data   = zipfile.ZipFile(data_path)
        
        self.data_name,self.max_len = zipToName(self.data)
        
        self.frequency  = frequency
        self.scale             = scale
        self.test              = test
        self.__dataset_information(label_path)
        
    def __dataset_information(self,label_path):
        
        '''
        Count how many videos are in the folder, map video names to each frames,
        map ints to video names
   
        Parameters
        ----------
        '''
        
        self.numbers_of_data    = 0
        
        
        with open(label_path) as f:
            labels = json.load(f)
        
        index = 0
        self.index_name_dic = dict()        
        for k,v in labels.items():
            if v[1] in self.fold:
                continue
            self.index_name_dic[index] = [k,v[0]]
            index +=  1
       
        self.numbers_of_data = index
                     
                  
    def __len__(self):
        
        '''
        return the length of videos
   
        Parameters
        ----------
        '''
        
        return self.numbers_of_data
    
    def __getitem__(self,idx):
        
        '''
        Given the int, load the correpsonding frames and labels
   
        Parameters
        ----------
        '''
        
        ids     =  self.index_name_dic[idx]
        name,label = ids
        y = np.zeros(1)
        y[0] = label
        pad_x,l = loadframes(self.data_name, self.data, name, self.frequency, self.max_len//(self.frequency[label])+2)
        return pad_x/self.scale,y,l
        
class SPOSDataGenerator(torch.utils.data.Dataset):
    
    '''
    The minibatch generater for SPOS database used when training the models
    
    Parameters
    ----------
    
    fold       : dict, that contains the map of ints to video
    frequency  : tuple,[int,int] the frequency used to sample the frames, which decide how many frames used to represent 
                 the videos. In the MMI databases, different fps are used. 
    test       : boolean, default false, whether it is the generator used for test
    scale      : the ratio used to scale the value of each frames
    
        
    Attributes
    ----------
    data_name      : dict, match the name of each videos to its frames
    number_of_data : int, the number of videos used
    max_len        : int, the maximum length of sampled videos
    index_name_dic : dict, map ints to the video name
    '''
    
    def __init__(self,fold, frequency = 5,test = False ,scale = 255):
        
        self.fold              = fold
        
        self.data_name = dict()
        
        for file in self.fold.values():
            for f in os.listdir(file):
                if not f.endswith(".bmp"):
                    continue
                if file in self.data_name:
                    self.data_name[file].append(f)
                else:
                    self.data_name[file] = [f]
        
        self.max_len = max([len(list(i)) for i in self.data_name.values()])//frequency + 2
        
        self.frequency  =  frequency
        
        self.scale = scale
        
        self.numbers_of_data = len(self.fold)
        
    def __len__(self):
        
        '''
        return the length of videos
   
        Parameters
        ----------
        '''

        
        return self.numbers_of_data
    
    def __getitem__(self,idx):
        
        '''
        Given the int, load the correpsonding frames and labels
   
        Parameters
        ----------
        '''

        name     =  self.fold[idx]
        y = np.zeros(1)
        
        if "spontaneous" in name:
            y[0] = 1
        
        values = []
        files = [f for f in self.data_name[name] if f.endswith(".bmp") and f.replace(".bmp","").isnumeric()]
        files = sorted(files, key = lambda x : int(x.replace(".bmp","")))

        last = [files[-1]] if len(files) % self.frequency >= 2/3 * self.frequency else []
        start = 0
        files = np.concatenate([np.array(files)[start::self.frequency],last])
        
        l  = 0
        
        pad_x = np.zeros((self.max_len,3,48,48)) #52
        for frame in files:
            cur  = Image.open(os.path.join(name,frame)).convert("RGB")
            cur = np.swapaxes(np.asarray(cur.resize((48,48))), 2, 0)
            values.append(cur)
            l+=1
            
        x = np.array(values)
        l = self.max_len - l
        pad_x[l:] = x
        return pad_x/self.scale,y,l    
