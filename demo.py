#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:03:10 2020

@author: Yan
"""

import argparse
import shutil
import os
from train import train_UVANEMO,train_SPOS,train_BBC,train_MMI


###############################################################
######## script for model training on different database ######
###############################################################
'''
Automatically run on GPU, if there is a one.
Note, only compatible with macos or linux.
No test conducted on Window
'''
parser = argparse.ArgumentParser(description='DeepSmileNet training')
parser.add_argument('--database', default="UVANEMO", type=str,
                    help='select the database to run, please selected from UVANEMO,SPOS,MMI and BBC')
parser.add_argument('--batch_size', default=16, type=int,
                    help='the mini batch size used for training')
parser.add_argument('--label_path', default=os.path.join("processed_data","label"), type=str,
                    help='the path contains training labels')
parser.add_argument('--frame_path', default=os.path.join("processed_data","uva.zip"), type=str,
                    help='the path contains processed data')
parser.add_argument('--frequency', default=5, type=int,
                    help='the frequency used to sample the data')
parser.add_argument('--sub', default="org", type=str,
                    help='the subsitution for the model, please selected from org,LSTM,GRU,resnet,miniAlexnet,minidensenet')
parser.add_argument('--epochs', default=10, type=int,help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='learning rate')




def main():
    args = parser.parse_args()
    
    if args.database == "UVANEMO":
        file_name = "uvanemo_training"
    if args.database == "SPOS": 
        file_name = "spos_training"
    if args.database == "BBC":
        file_name = "bbc_training"
    if args.database == "mmi": 
        file_name = "mmi_training"   
        
    try :
        shutil.rmtree(f"{file_name}")
        os.makedirs(f"{file_name}")
    except FileNotFoundError:
        os.makedirs(f"{file_name}")
        
    if args.database == "UVANEMO":
        train_UVANEMO(args.epochs,args.lr,args.label_path, args.frame_path,args.frequency, args.batch_size,args.sub)
    if args.database == "SPOS":
        train_SPOS(args.epochs,args.lr,args.frame_path,args.frequency, args.batch_size,args.sub)
    if args.database == "BBC":
        train_BBC(args.epochs,args.lr,args.frame_path,args.frequency, args.batch_size,args.sub)
    if args.database == "MMI":
        train_MMI(args.epochs,args.lr,args.frame_path,args.frequency, args.batch_size,args.sub)
    
    
if __name__ == '__main__':
    main()