import os
import shutil
import torch
import numpy as np
import random
import time
import copy
import glob, glob2
from torch import nn

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class MLP_dict_softmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,
                 edge_types=10):
        super(MLP_dict_softmax, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim=input_dim, output_dim=self.bottleneck_dim, hidden_size=hidden_size)
        # self.dict_layer = conv1x1(self.bottleneck_dim,output_dim)
        # self.dict_layer = nn.Linear(self.bottleneck_dim,output_dim,bias=False)
        self.MLP_factor = MLP(input_dim=input_dim, output_dim=1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = gumbel_softmax(self.MLP_distribution(x), tau=1 / 2, hard=False)
        # embed = self.dict_layer(distribution)
        factor = torch.sigmoid(self.MLP_factor(x))
        # factor = 1
        out = factor * distribution
        return out, distribution


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout / 3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class MLP_dict(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,
                 edge_types=10):
        super(MLP_dict, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim=input_dim, output_dim=self.bottleneck_dim, hidden_size=hidden_size)
        self.MLP_factor = MLP(input_dim=input_dim, output_dim=1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = torch.abs(self.MLP_distribution(x))
        return distribution, distribution



