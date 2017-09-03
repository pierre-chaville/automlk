# automlk
Automated machine learning toolkit
----------------------------------

This toolkit is designed to be integrated within a python project, but also independenly through the interface of the app.

The framework is designed with similar principles than auto-sklearn, with the following improvements:
- web interface (flask) to review the datasets, the search results and graphs
- include sklearn models, but also Xgboost, LightGBM, CatBoost and keras Neural Networks
- 2nd level ensembling with model selection and stacking
- can be used in competion mode (to generate a submit file from a test set), on public mode (separate train set and public set) and standard mode.

We have provided some public datasets to initialize the framework and compare results with best scores.

Usage:
-----
- create a dataset
- launch the search in auto mode: this will search the best pre-processing steps, machine learning models and ensembles
- view the results through with the web interface

References:
----------
Feurer, Matthias, et al. "Efficient and robust automated machine learning." Advances in Neural Information Processing Systems. 2015.
