Installation
============

Pre-requisites
--------------

**Sklearn version must be > 0.19, otherwise there will be several blocking issues.**

to upgrade scikit-learn:

*On conda:*

.. code-block:: python

    conda update conda

    conda update scikit-learn

*If you do not use conda, update with pip:*

.. code-block:: python

    pip install scikit-learn --update


**Warning: if you use conda, you must absolutely update sklearn with conda**

Additionally, you must also install category_encoders:

.. code-block:: python

    pip install category_encoders

Optionally, you may install the following models:

* LightGBM (highly recommended, because it is very quick and efficient):

.. code-block:: python

    pip install lightgbm

* Xgboost (highly recommended, because it is also state of the art):

*See Xgboost documentation for installation*

* Catboost:

.. code-block:: python

    pip install catboost

* keras with theano or tensorflow:

*See keras, theano or tensorflow documentation for installation*

Installation
------------

Download the module from github and extract the zip file in a folder (by default automlk-master)

Install as:

.. code-block:: python

    cd automlk-master

    python setup.py install


Basic installation
------------------

The simplest installation runs on a single machine, with at least the following processes:
1. the web app
2. the controller
3. a single worker

These 3 components are run in a console (Windows) or Terminal (Linux).

The basic installation will use a data folder on the same machine.
By default, the data folder should be created at one level upper the automlk-master folder.

For example, let's assume that autoMLk is created in the $HOME (Linux) level or Documents (windows):

* home
    - pierre
        * automlk-master
            - automlk
            - run
            - web
        * data

If you want to use a data folder in another location, you can define this in the config screen.

To run the web app:

.. code-block:: python

    cd automlk-master/web

    python run.py

This will launch the web app, which can be accessed from a web browser, at the following address:

.. code-block:: python

    http://localhost:5001

From the web app, you can now define the set-up and then import the example of datasets.

You can launch the search in a dataset simply by clicking on the start/pause button in the home screen, and view the results through with the web interface.
The search will continue automatically until the search is completed.

To run the controller:

.. code-block:: python

    cd automlk-master/run

    python run_controller.py

To run the worker:

*On Linux:*

.. code-block:: python

    cd automlk-master/run

    sh worker.sh

*On Windows:*

.. code-block:: python

    cd automlk-master/run

    worker

Note:
This will run the python module ru_worker.py in an infinite loop, in order to catch the potential crashes from the worker.

Advanced configuration
----------------------

