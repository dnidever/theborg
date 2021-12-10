***************
Getting Started
***************



How it works
============

|theborg| has routines to train Artificial Neural Networks (ANNs).  The two types currently supported
are ``classifier`` and ``emulator``.  The models and training recipes are based on software written by Yuan-Sen Ting:
`The Payne <https://github.com/tingyuansen/The_Payne>`_ for the emulator and the `MWM Classifier <https://github.com/tingyuansen/MWM_Classifier>`_ for the convolutional classifier.


|theborg| Model Basics
======================

Every |theborg| model can do a few important things:

 - :meth:`~theborg.model.Model.train`: This trains the model on a training set.
 - :meth:`call <theborg.model.Model>`: Run the emulator for a set of data/labels, e.g., ``out = model(input)``.
 - :meth:`~theborg.model.Model.save`: Save the model to a file.
 - :meth:`~theborg.model.Model.load`: Load a model from a file.

Training a Model
================
   
We use a training set with ``data`` and ``labels`` to train the models.
The dimensions of the training_labels and training_data should be:

 - training_labels: [# training data, # labels]
 - training_data: [# training data, # features]

You can specify various parameters of the model.  For example, ``num_neurons``, the number of
neurons in the hidden layer(s).  You can also specify some training parameters, such as how
quickly the model learns (``learning rate``), the chunks of training data that are fed through
the model simultaneously during the training process (``batch size``), and the number of iterations
to train (``num_steps``).

It is substantially faster to train an ANN using a GPU with the ``cuda=True`` setting.  By default,
``cuda=True`` so a model can be trained on a regular CPU.


Classifier
==========

A classifier takes data with various ``features`` and returns a probability of it being a member of
various classes.  

.. code-block:: python

        from theborg.classifier import Classifier
	cmodel = Classifier()
	cmodel.train(training_labels=labels,training_data=data)
	probs = cmodel(data)

The input data when calling the model will have dimensions of [# of features] and the output will
have probabilities [# of labels] that sum to one.
	
Now you can save the model to a file.
	
.. code-block:: python

	cmodel.save('my_classifier.pkl')

There are two classifiers.  A simple classifier :class:`~theborg.classifier.Classifer` using a serial model, and
a classifier using convolutional layers :class:`~theborg.classifier.CNNClassifer`.
	
	
Emulator
========

An ``emulator`` simulates real data using only a few labels.  For example, a stellar spectrum
is simulated using just temperature, surface gravity and metallicity. :class:`~theborg.emulator.Emulator` can be
used to emulate a wide variety of data.

.. code-block:: python

        from theborg.emulator import Emulator
	emodel = Emulator()
	emodel.train(training_labels=labels,training_data=data)
	spec = emodel(labels)

The input data when calling the model will have dimensions of [# of labels] and the output will
have dimensions of [# of features].

