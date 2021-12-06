# import package
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import torch
import time
from . import radam


#===================================================================================================
# convolutional models
# simple multi-layer perceptron model
class ClassifierModel(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_channels, mask_size, stride_size, num_group):
        super(classifier, self).__init__()

        self.conv1 = torch.nn.Sequential(
                       torch.nn.Conv1d(1, num_channels, mask_size),
                       torch.nn.MaxPool1d(kernel_size=mask_size, stride=stride_size),
                       torch.nn.LeakyReLU()
        )

        self.conv2 = torch.nn.Sequential(
                       torch.nn.Conv1d(num_channels, num_channels, mask_size),
                       torch.nn.MaxPool1d(kernel_size=mask_size, stride=stride_size),
                       torch.nn.LeakyReLU()
        )

        self.conv3 = torch.nn.Sequential(
                       torch.nn.Conv1d(num_channels, num_channels, mask_size),
                       torch.nn.MaxPool1d(kernel_size=mask_size, stride=stride_size),
                       torch.nn.LeakyReLU()
        )
        self.conv4 = torch.nn.Sequential(
                       torch.nn.Conv1d(num_channels, 1, mask_size),
                       torch.nn.MaxPool1d(kernel_size=mask_size, stride=stride_size),
                       torch.nn.LeakyReLU()
        )

        # calculate number of features after convolution
        num_features = dim_in
        for i in range(4):
            num_features = (num_features-mask_size) + 1 # from covolution # no stride
            num_features = (num_features-mask_size)//stride_size + 1 # from max pooling

        print("Number of features before the dense layers:", num_features)

        self.features = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_group),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.features(x[:,0,:])
        return x

#===================================================================================================
# train neural networks
def neural_net(training_spectra, training_labels, validation_spectra, validation_labels,\
               wavelength_note = "IR",
               num_channels=8, num_neurons = 30, mask_size=11, stride_size=3,\
               num_steps=1e3, learning_rate=1e-4, batch_size=256):

    '''
    Training neural networks as a spectra broker
    '''

    # dimension of the input
    dim_in = training_spectra.shape[1]
    num_group = training_labels.shape[1]

#------------------------------------------------------------------------------
    # make pytorch variables
    x = torch.from_numpy(training_spectra).type(torch.cuda.FloatTensor)
    y = torch.from_numpy(training_labels).type(torch.cuda.FloatTensor)
    x_valid = torch.from_numpy(validation_spectra).type(torch.cuda.FloatTensor)
    y_valid = torch.from_numpy(validation_labels).type(torch.cuda.FloatTensor)

    # run on cuda
    x.cuda()
    y.cuda()
    x_valid.cuda()
    y_valid.cuda()

    # expand into 3D (i.e. 1 channel)
    x = x[:,None,:]
    x_valid = x_valid[:,None,:]

#--------------------------------------------------------------------------------------------
    # assume cross entropy loss
    loss_fn = torch.nn.BCELoss()

    # initiate the classifier
    model = classifier(dim_in=dim_in, num_neurons=num_neurons, num_channels=num_channels,\
                       mask_size=mask_size, stride_size=stride_size, num_group=num_group)
    model.cuda()
    model.train()

    # we adopt rectified Adam for the optimization
    optimizer = radam.RAdam([p for p in model.parameters() if p.requires_grad==True], lr=learning_rate)

#--------------------------------------------------------------------------------------------
    # break into batches
    nsamples = x.shape[0]
    nbatches = nsamples // batch_size

    nsamples_valid = x_valid.shape[0]
    nbatches_valid = nsamples_valid // batch_size

    # initiate counter
    current_loss = np.inf
    training_loss =[]
    validation_loss = []

#-------------------------------------------------------------------------------------------------------
    # train the network
    for e in range(int(num_steps)):

        # randomly permute the data
        perm = torch.randperm(nsamples)
        perm = perm.cuda()

        # for each batch, calculate the gradient with respect to the loss
        for i in range(nbatches):
            idx = perm[i * batch_size : (i+1) * batch_size]
            y_pred = model(x[idx])
            loss = loss_fn(y_pred, y[idx])*1e4
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        # the average loss
        if e % 10 == 0:

            # randomly permute the data
            perm_valid = torch.randperm(nsamples_valid)
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

            # record the weights and biases if the validation loss improves
            if loss_valid_data < current_loss:
                current_loss = loss_valid_data

                state_dict =  model.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu()
                torch.save(state_dict, '../NN_normalized_spectra_' + wavelength_note + '.pt')

                np.savez("../training_loss" + wavelength_note + ".npz",\
                         training_loss = training_loss,\
                         validation_loss = validation_loss)

#--------------------------------------------------------------------------------------------
    # save the final training loss
    np.savez("../training_loss_" + wavelength_note + ".npz",\
             training_loss = training_loss,\
             validation_loss = validation_loss)
