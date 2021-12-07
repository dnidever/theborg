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
from torch.autograd import Variable
from . import radam

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

class Emulator(object):
    def __init__(self, dim_in=4, num_neurons=100, num_features=500, training_data=None, training_labels=None):
        self.model = EmulatorModel(dim_in, num_neurons, num_features)
        self.trained = False
        self.training_loss = []
        self.validation_loss = []
        self.training_data = training_data
        self.training_labels = training_labels
        
    def __call__(self,labels):
        """ Return the model value."""
        # Input labels should be unscaled
        data = self._data
        # assuming your NN has two hidden layers.
        x_min, x_max = data['x_min'],data['x_max']
        scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5   # scale the labels
        
        w_array_0, w_array_1, w_array_2 = data['w_array_0'],data['w_array_1'],data['w_array_2']
        b_array_0, b_array_1, b_array_2 = data['b_array_0'],data['b_array_1'],data['b_array_2']            

        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        model = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2

        # maybe use model.forward()
        # need to input torch tensor variable values
        #dtype = torch.FloatTensor
        #x = Variable(torch.from_numpy(np.array(labels))).type(dtype)  
        #out = self.model.forward(x)
        
        return model

    def write(self,outfile):
        """ Write the model to a file."""
        # save parameters and remember how we scaled the labels
        np.savez(outfile,
                 w_array_0 = self._data['w_array_0'],
                 b_array_0 = self._data['b_array_0'],
                 w_array_1 = self._data['w_array_1'],
                 b_array_1 = self._data['b_array_1'],
                 w_array_2 = self._data['w_array_2'],
                 b_array_2 = self._data['b_array_2'],
                 x_max = self._data['x_max'],
                 x_min = self._data['x_min'],
                 num_labels = self._data['num_labels'],
                 num_features = self._data['num_features'],
                 learning_rate = self._data['learning_rate'],
                 num_neurons = self._data['num_neurons'],
                 num_steps = self._data['num_steps'],
                 batch_size = self._data['batch_size'],
                 labels = self._data['labels'],
                 training_loss = self.training_loss,
                 validation_loss = self.validation_loss)

    @classmethod
    def read(cls,infile):
        """ Read the model from a file."""
        temp = np.load(infile)
        model_data = {}
        for f in temp.files:
            model_data[f] = temp[f]
        mout = Emulator(model_data['num_labels'], model_data['num_neurons'], model_data['num_features'])
        mout.trained = True
        mout._data = model_data
        return mout
        
    #===================================================================================================
    # train neural networks
    def train(self,training_labels, training_data, validation_labels=None, validation_data=None,
              validation_split=0.2, num_neurons=None, num_steps=1e4, learning_rate=1e-4, batch_size=200,
              num_features=None, cuda=False, name=None,shuffle=True,
              label_names=None):

        '''
        Training a neural net to emulate spectral models

        training_labels has the dimension of [# training spectra, # stellar labels]
        training_data has the dimension of [# training spectra, # wavelength pixels]
        
        The validation set is used to independently evaluate how well the neural net
        is emulating the spectra. If the neural network overfits the spectral variation, while
        the loss will continue to improve for the training set, but the validation
        set should show a worsen loss.

        The training is designed in a way that it always returns the best neural net
        before the network starts to overfit (gauged by the validation set).

        num_steps = how many steps to train until convergence.
        1e4 is good for the specific NN architecture and learning I used by default.
        Bigger networks will take more steps to converge, and decreasing the learning rate
        will also change this. You can get a sense of how many steps are needed for a new
        NN architecture by plotting the loss evaluated on both the training set and
        a validation set as a function of step number. It should plateau once the NN
        has converged.

        learning_rate = step size to take for gradient descent
        This is also tunable, but 1e-4 seems to work well for most use cases. Again,
        diagnose with a validation set if you change this.

        num_features is the number of features before the deconvolutional layers; it only
        applies if ResNet is used. For the simple multi-layer perceptron model, this parameter
        is not used. We truncate the predicted model if the output number of pixels is
        larger than what is needed. In the current default model, the output is ~8500 pixels
        in the case where the number of pixels is > 8500, increase the number of features, and
        tweak the ResNet model accordingly

        batch_size = the batch size for training the neural networks during the stochastic
        gradient descent. A larger batch_size reduces stochasticity, but it might also
        risk of stucking in local minima
        
        '''

        # Output names
        if name is None:
            modelfile = "NN_normalized_data.npz"
            lossfile = "trained_loss.npz"
        else:
            modelfile = "NN_normalized_data_"+str(name)+".npz"
            lossfile = "trained_loss_"+str(name)+".npz"
        
        # run on cuda
        if cuda:
            dtype = torch.cuda.FloatTensor
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            dtype = torch.FloatTensor
            torch.set_default_tensor_type('torch.FloatTensor')   

        # Default label names
        if label_names is None:
            label_names = []
            for i in range(len(num_labels)):
                label_names.append('label'+str(i+1))

        # Shuffle the data
        ndata,num_features = training_data.shape
        ndata2,num_labels = training_labels.shape
        ind = np.arange(ndata)
        if shuffle:
            np.random.shuffle(ind)  # shuffle in place
            
        # Validation split
        if validation_labels is None and validation_data is None and validation_split is not None:
            vsi = np.arange(ndata)
            np.random.shuffle(vsi)   # shuffle
            vsi = vsi[0:int(np.round(validation_split*ndata))]  # only want validation_split
            vind = ind[vsi]
            ind = np.delete(ind,vsi)   # remove these from the training set
            validation_data = training_data[vind,:] 
            validation_labels = training_labels[vind,:]

        # Default num_neurons
        if num_neurons is None:
            num_neurons = 2*num_features
            print('num_neurons not input.  Using 2*Nfeatures = ',num_neurons)
            
        # Re-initialize the model and trained data and history
        self.model = EmulatorModel(num_labels, num_neurons, num_features)
        self._data = None
        self.training_loss = []
        self.validation_loss = []
        self.trained = False
        
        # scale the labels, optimizing neural networks is easier if the labels are more normalized
        x_max = np.max(training_labels[ind,:], axis = 0)
        x_min = np.min(training_labels[ind,:], axis = 0)
        x = (training_labels[ind,:] - x_min)/(x_max - x_min) - 0.5
        x_valid = (validation_labels-x_min)/(x_max-x_min) - 0.5

        # dimension of the input
        dim_in = x.shape[1]

        #--------------------------------------------------------------------------------------------
        # assume L1 loss
        loss_fn = torch.nn.L1Loss(reduction = 'mean')

        # make pytorch variables
        x = Variable(torch.from_numpy(x)).type(dtype)
        y = Variable(torch.from_numpy(training_data[ind,:]), requires_grad=False).type(dtype)
        x_valid = Variable(torch.from_numpy(x_valid)).type(dtype)
        y_valid = Variable(torch.from_numpy(validation_data), requires_grad=False).type(dtype)

        # initiate EmulatorModel and optimizer
        model = self.model
        if cuda:
            model.cuda()
        model.train()

        # we adopt rectified Adam for the optimization
        optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

        #--------------------------------------------------------------------------------------------
        # train in batches
        nsamples = x.shape[0]
        nbatches = nsamples // batch_size

        nsamples_valid = x_valid.shape[0]
        nbatches_valid = nsamples_valid // batch_size
        
        # initiate counter
        current_loss = np.inf
        training_loss = []
        validation_loss = []

        #-------------------------------------------------------------------------------------------------------
        # train the network
        for e in range(int(num_steps)):

            # randomly permute the data
            perm = torch.randperm(nsamples)
            if cuda:
                perm = perm.cuda()

            # for each batch, calculate the gradient with respect to the loss
            for i in range(nbatches):
                idx = perm[i * batch_size : (i+1) * batch_size]
                y_pred = model(x[idx])

                loss = loss_fn(y_pred, y[idx])*1e4
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()

            #-------------------------------------------------------------------------------------------------------
            # evaluate validation loss
            if e % 100 == 0:

                # here we also break into batches because when training ResNet
                # evaluating the whole validation set could go beyond the GPU memory
                # if needed, this part can be simplified to reduce overhead
                perm_valid = torch.randperm(nsamples_valid)
                if cuda:
                    perm_valid = perm_valid.cuda()
                loss_valid = 0

                for j in range(nbatches_valid):
                    idx = perm_valid[j * batch_size : (j+1) * batch_size]
                    y_pred_valid = model(x_valid[idx])
                    loss_valid += loss_fn(y_pred_valid, y_valid[idx])*1e4
                loss_valid /= nbatches_valid

                print('iter %s:' % e, 'training loss = %.3f' % loss,\
                      'validation loss = %.3f' % loss_valid)

                loss_data = loss.detach().data.item()
                loss_valid_data = loss_valid.detach().data.item()
                training_loss.append(loss_data)
                validation_loss.append(loss_valid_data)

                #--------------------------------------------------------------------------------------------
                # record the weights and biases if the validation loss improves
                if loss_valid_data < current_loss:
                    current_loss = loss_valid_data
                    model_numpy = []
                    for param in model.parameters():
                        model_numpy.append(param.data.cpu().numpy())

                    # extract the weights and biases
                    model_data = {}
                    model_data['w_array_0'] = model_numpy[0]
                    model_data['b_array_0'] = model_numpy[1]
                    model_data['w_array_1'] = model_numpy[2]
                    model_data['b_array_1'] = model_numpy[3]
                    model_data['w_array_2'] = model_numpy[4]
                    model_data['b_array_2'] = model_numpy[5]
                    model_data['x_min'] = x_min
                    model_data['x_max'] = x_max
                    model_data['num_labels'] = num_labels
                    model_data['num_features'] = num_features                    
                    model_data['learning_rate'] = learning_rate
                    model_data['num_neurons'] = num_neurons
                    model_data['num_steps'] = num_steps
                    model_data['batch_size'] = batch_size
                    model_data['labels'] = label_names
                    model_data['niter'] = e
                    model_data['training_loss'] = loss_data
                    model_data['validation_loss'] = loss_valid_data

                    self._data = model_data
                    self.training_loss = training_loss                    
                    self.validation_loss = validation_loss


        #--------------------------------------------------------------------------------------------
        # Final values
        # extract the weights and biases
        model_data = {}
        model_data['w_array_0'] = model_numpy[0]
        model_data['b_array_0'] = model_numpy[1]
        model_data['w_array_1'] = model_numpy[2]
        model_data['b_array_1'] = model_numpy[3]
        model_data['w_array_2'] = model_numpy[4]
        model_data['b_array_2'] = model_numpy[5]
        model_data['x_min'] = x_min
        model_data['x_max'] = x_max
        model_data['num_labels'] = num_labels
        model_data['num_features'] = num_features                    
        model_data['learning_rate'] = learning_rate
        model_data['num_neurons'] = num_neurons
        model_data['num_steps'] = num_steps
        model_data['batch_size'] = batch_size
        model_data['labels'] = label_names
        model_data['niter'] = e
        model_data['training_loss'] = loss_data
        model_data['validation_loss'] = loss_valid_data

        self._data = model_data
        self.training_loss = training_loss                    
        self.validation_loss = validation_loss
        self.trained = True
