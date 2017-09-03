DataSet
=======

The features of the automated machine learning are defined and stored in the DataSet object.
All features and data of a DataSet object can be viewed with the web app.

We have included a sample of public datasets to start with autoMLk.

To use these datasets, please run the dataset_loader program located in examples

.. code-block:: python

    cd automlk/examples
    python dataset_loader.py

the data describing these datasets are located in the csv file 'dataset.csv' in the automlk/datasets folder.
You may use the same format to create your own datasets.

.. autoclass:: automlk.dataset.DataSet
    :members:

.. autofunction:: automlk.dataset.get_dataset_list

.. autofunction:: automlk.dataset.get_dataset