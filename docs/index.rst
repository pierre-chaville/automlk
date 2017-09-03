..  autoMLk documentation master file, created by
    sphinx-quickstart on Sun Sep  3 19:15:35 2017.


AutoMLk: automated machine learning toolkit
===========================================

This toolkit is designed to be integrated within a python project, but also independenly through the interface of the app.

The framework is designed with similar principles than auto-sklearn, with the following improvements:

* web interface (flask) to review the datasets, the search results and graphs
* include sklearn models, but also Xgboost, LightGBM, CatBoost and keras Neural Networks
* 2nd level ensembling with model selection and stacking
* can be used in competion mode (to generate a submit file from a test set), on public mode (separate train set and public set) and standard mode.


.. figure:: img/best.png
   :scale: 50 %
   :alt: models with the best scores

   Best models by eval score

We have provided some public datasets to initialize the framework and compare results with best scores.

Usage
-----

install as:

.. code-block:: python

    python setup.py install

* create a dataset
* launch the search in auto mode: this will search the best pre-processing steps, machine learning models and ensembles
* view the results through with the web interface

Content
-------

.. toctree::
    :maxdepth: 3

    dataset
    searching
    app


Indices
-------

* :ref:`genindex`


