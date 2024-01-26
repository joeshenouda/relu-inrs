#!/usr/bin/env python

import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class Bspline(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.name = 'bspline'

    def forward(self, x):
        return torch.as_tensor(F.relu(x) - 2*F.relu(x-torch.as_tensor(1)) + F.relu(x-torch.as_tensor(2)))

class BsplineLayer(nn.Module):
    '''
        Implicit representation with B-spline nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
    '''
    
    def __init__(self, in_features, out_features, bias=True, is_first=False, coords=None):
        super().__init__()
        
        self.in_features = in_features
        self.is_first = is_first
        
        self.act_func = Bspline()
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias)
            
    
    
    def forward(self, input):
        x = self.linear(input)
        
        return self.act_func(x)

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, out_features,first_omega_0, hidden_omega_0,
                        scale, pos_encode, sidelength, fn_samples, use_nyquist, outermost_linear=True, 
                        skip_conn=True):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
 
        self.nonlin = BsplineLayer
        
        self.wavelet = 'not-wavelet'    
                    
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, is_first=True))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features))

        final_linear = nn.Linear(hidden_features,
                                 out_features)            
        
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
        if skip_conn:
            self.skip_conn = True
            self.skip_connection = nn.Linear(in_features, out_features)
        else:
            self.skip_conn = False
        
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
        
        if self.skip_conn:
            output = output + self.skip_connection(coords)
         
        return output