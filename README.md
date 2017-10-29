# automlk
Automated and distributed machine learning toolkit
--------------------------------------------------

This toolkit is designed to be integrated within a python project, but also independently through the interface of the app.

The framework is designed with similar principles than auto-sklearn, with the following improvements:
- **web interface** (flask) to review the datasets, the search results and graphs
- **distributed architecture** with multiple search workers on multiple machines
- can also be run on a single machine
- include sklearn models, but also **Xgboost, LightGBM, CatBoost and keras Neural Networks***
- 2nd level ensembling with model selection and **stacking**
- can be used in competition mode (to generate a submit file from a test set), on public mode (separate train set and public set) and standard mode.
- **pre-processing** hyper-optimization in parallel with model hyper-parameter optimization
- **text pre-processing** on specific text columns with bow, word2vec, doc2vec, ...
- **automatic documentation of the models in html and pdf format**

We have provided some public datasets to initialize the framework and compare results with best scores.

[Find the documentation here](http://automlk.readthedocs.io/en/latest/)

*This framework is currently an alpha version*

Installation
------------
download and then install:

    python setup.py install

then run the web server to define the set-up and launch the workers ([see documentation](http://automlk.readthedocs.io/en/latest/))

Usage
-----
Launch the web app in /web folder:

    python run.py

This will launch the web app, which can be accessed via a web browser, at address:

    http://localhost:5001

From the web app, you can now import the example of datasets (import in the menu, then select the dataset.csv in the /data folder)

You can launch the search in a dataset simply by clicking on the |> button in the home screen, and view the results through with the web interface.
The search will continue automatically until the search is completed.


Requirements
------------
- category_encoders

optional:
- lightGBM
- Xgboost
- Catboost
- Keras with Theano or Tensorflow (Neural Networks)
- Gensim (word2vec, doc2vec)

- Redis (for in memory key/value storage and queues)

References
----------
Feurer, Matthias, et al. "Efficient and robust automated machine learning." Advances in Neural Information Processing Systems. 2015.
