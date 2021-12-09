'''
This file is used to train the neural net that predicts the spectrum
given any set of stellar labels (stellar parameters + elemental abundances).

Note that, the approach here is slightly different from Ting+19. Instead of
training individual small networks for each pixel separately, here we train a single
large network for all pixels simultaneously.

The advantage of doing so is that individual pixels will exploit information
from adjacent pixels. This usually leads to more precise interpolations.

However to train a large network, GPU is needed. This code will
only run with GPU. But even with an inexpensive GPU, this code
should be pretty efficient -- training with a grid of 10,000 training spectra,
with > 10 labels, should not take more than a few hours

The default training set are the Kurucz synthetic spectral models and have been
convolved to the appropriate R (~22500 for APOGEE) with the APOGEE LSF.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import torch
import time
import dill as pickle
from collections import OrderedDict
from torch.autograd import Variable
from . import radam
from .model import Model

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''

    return z*(z > 0) + 0.01*z*(z < 0)


#===================================================================================================
# simple multi-layer perceptron model
class EmulatorModel(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features):
        super(EmulatorModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_features),
        )

    def forward(self, x):
        return self.features(x)


#===================================================================================================
# simple multi-layer perceptron model

class Emulator(Model):
    def __init__(self, dim_in=4, num_neurons=100, num_features=500, **kwargs):
        super().__init__(dim_in, num_neurons, num_features, **kwargs)
        self.model = EmulatorModel(dim_in, num_neurons, num_features)
