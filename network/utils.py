#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:25:40 2019

@author: tony
"""

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def accuracy(out, labels):
    total = 0.0
    _,predicted = torch.max(out, 1)
    size = len(predicted)
    labels= torch.argmax(labels.data)
    total += torch.sum(predicted == labels.data)
    return total.cpu().detach().numpy()/size