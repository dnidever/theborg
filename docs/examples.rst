********
Examples
********


Running |theborg|
=================

With a training set you can simple train an emulator.

.. code-block:: python

        from theborg.emulator import Emulator
	emodel = Emulator()
	emodel.train(training_labels=labels,training_data=data)
	spec = emodel(labels)
