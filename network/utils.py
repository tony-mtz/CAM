#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:25:40 2019

@author: tony
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


def accuracy(input:Tensor, targs:Tensor):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean().cpu().detach().numpy()

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def imshow_transform(image_in, title=None):
    """Imshow for Tensor."""
    img = np.rollaxis(image_in.squeeze().cpu().detach().numpy(),0,3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    return img

#https://www.fast.ai/
#fastai code snippet
class SaveFeatures():
    features=None
    def __init__(self,m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
