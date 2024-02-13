#!/usr/bin/env python

import os
import sys
import tqdm
import ipdb
import collections

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import torch.nn.utils.weight_norm as w_norm

class BSplineWavelets(nn.Module):
    def __init__(self,  scale=1):
        super().__init__()
        self.name = 'bspline-w'
        self.c = scale

    def forward(self, x):
        #shift = 1.5
                
        # The shift is to make it symmetric about the y-axis
        #x = self.c * x + shift
        x = self.c * x

        term1 = (1 / 6) * F.relu(x)
        term2 = - (8 / 6) * F.relu(x - (1 / 2))
        term3 = (23 / 6) * F.relu(x - (1))
        term4 = - (16 / 3) * F.relu(x - (3 / 2))
        term5 = (23 / 6) * F.relu(x - (2))
        term6 = - (8 / 6) * F.relu(x - (5 / 2))
        term7 = (1 / 6) * F.relu(x - (3))

        sum = term1 + term2 + term3 + term4 +  term5 + term6 + term7

        return sum


class BsplineWaveletLayer(nn.Module):
    '''
        B-spline wavelet nonlinearity layer
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Set to Trueif first layer of the network.
            is_last: Set to True if last layer of the network.
            weight_norm: Set to True to use weight normalization parameterization (slightly speeds things up).
            init_w: Manually initialize the weights.
            init_scale: Initial scaling of the weights.
            linear_layers: If True parameterizes network with linear layers in between.
            hidden_features: Number of neurons in this layer.
    '''
    
    def __init__(self, in_features, out_features, bias=True, weight_norm=False, c=1, init_w = None, init_scale=1, linear_layer=False):
        
        super().__init__()
        
        self.in_features = in_features
        self.c = c
        self.linear_layer = linear_layer
        self.weight_norm = weight_norm

        self.wavelon = []
                
        if weight_norm:
            lin_layer_W = w_norm(nn.Linear(in_features,
                                out_features,
                                bias=bias))
            lin_layer_W.weight.data = init_scale * lin_layer_W.weight.data        
        else:
            lin_layer_W = nn.Linear(in_features,
                                    out_features,
                                    bias=bias)
            
            lin_layer_W.weight.data = init_scale * lin_layer_W.weight.data
            
            if init_w:
                lin_layer_W.weight.data = init_w
        
        self.wavelon.append(('linear',lin_layer_W))
        self.wavelon.append(('bspline_w', BSplineWavelets(scale=c)))


        if self.linear_layer:
            lin_layer_V = nn.Linear(out_features, 
                                 out_features, 
                                 bias=bias)
            lin_layer_V.weight.data = init_scale * lin_layer_V.weight.data
            self.wavelon.append(('linear2',lin_layer_V))

        self.block= torch.nn.Sequential(collections.OrderedDict(self.wavelon))

            
    
    
    def forward(self, input):
        out = self.block(input)
        
        return out

class INR(nn.Module):
    def __init__(self, in_features, 
                 hidden_features, 
                 hidden_layers, 
                 out_features,
                 first_omega_0, 
                 hidden_omega_0,
                 scale, 
                 pos_encode, 
                 sidelength, 
                 fn_samples, 
                 use_nyquist, 
                 outermost_linear=True, 
                 skip_conn=False, 
                 resnet=False, 
                 weight_norm=False, 
                 init_scale=1,
                 linear_layers=False):
        
        super().__init__()
        
        if resnet:
            self.nonline = BsplineWaveletResNet
        else:
            self.nonlin = BsplineWaveletLayer
        
        self.wavelet = 'bspline'    
                    
        self.net = []

        if hidden_layers == 1:
            self.net.append(self.nonlin(in_features,
                                        hidden_features,
                                        weight_norm=weight_norm,  
                                        c=first_omega_0, init_scale=init_scale,
                                        linear_layer=False))

            # Final layer
            self.net.append(torch.nn.Linear(hidden_features, out_features))
            

        if hidden_layers > 1:
            self.net.append(self.nonlin(in_features,
                                        hidden_features, weight_norm=weight_norm,
                                        c=hidden_omega_0, 
                                        init_scale=init_scale, linear_layer=linear_layers))

            for i in range(hidden_layers-2):
                self.net.append(self.nonlin(hidden_features,
                                        hidden_features, weight_norm=weight_norm,
                                        c=hidden_omega_0, 
                                        init_scale=init_scale,  linear_layer=linear_layers))
            
            final_layer = self.nonlin(hidden_features,
                                        hidden_features, weight_norm=weight_norm,
                                        c=hidden_omega_0, 
                                        init_scale=init_scale,  linear_layer=False)
            
            final_layer.block.add_module('linear2',nn.Linear(hidden_features, out_features))
            self.net.append(final_layer)


            
        self.net = nn.Sequential(*self.net)
        
        if skip_conn and not resnet:
            self.skip_conn = True
            self.skip_connection = nn.Linear(in_features, out_features)
        else:
            self.skip_conn = False
        
    
    def forward(self, coords):
        output = self.net(coords)
        
        if self.skip_conn:
            output = output + self.skip_connection(coords)
         
        return output