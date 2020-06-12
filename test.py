#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:04:25 2020

@author: Yan
"""

import unittest
import torch
from model import DeepSmileNet


##################################
######## Basic Test Program ######
##################################

#create test input
test_input  = torch.zeros(2,10,3,48,48)
test_length = torch.LongTensor([2,2]) 
    

class TestDeepSmileNet(unittest.TestCase):
    
    #test different component of DeepSmileNet
    def test_model(self):
        
        for re in ["org","LSTM","GRU","resnet","miniAlexnet","minidensenet"]:
            m = DeepSmileNet(re = re)
            self.assertEqual(m(test_input,test_length).size(0),2,"should be 2")
            self.assertEqual(m(test_input,test_length).size(1),1,"should be 1")
            
    def branches_model(self):
        
        m = DeepSmileNet(re = "org")
        self.assertEqual(m.decoder.cell.conv.weight.size(0),32,"should be 32")
            
        for re in ["resnet","miniAlexnet","minidensenet"]:
            m = DeepSmileNet(re = re)
            self.assertEqual(m.decoder.cell.conv.weight.size(0),128,"should be 128")
            
        m = DeepSmileNet(re = "GRU")
        self.assertEqual(m.decoder.weight_hh_l0.size(0),768,"should be 768")
        self.assertEqual(m.decoder.weight_ih_l0.size(0),768,"should be 768")
        
        m = DeepSmileNet(re = "LSTM")
        self.assertEqual(m.decoder.weight_hh_l0.size(0),1024,"should be 1024")
        self.assertEqual(m.decoder.weight_ih_l0.size(0),1024,"should be 1024")
        
if __name__ == '__main__':
    unittest.main()