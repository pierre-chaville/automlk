from .hyper import *

"""
this module defines the hyper-parameter spaces for the various models
"""
# generic parameters for tree models
default_sklearn_trees = {'n_estimators': 50, 'verbose': 0, 'random_state': 0}
space_sklearn_trees = {'n_estimators': HyperWeights({HyperRangeInt(10, 100): 2,
                                                     HyperRangeInt(100, 500): 3,
                                                     HyperRangeInt(500, 1000): 1}),
                       'max_features': HyperWeights({'auto': 2,
                                                     'sqrt': 1,
                                                     'log2': 1,
                                                     None: 1,
                                                     HyperRangeFloat(0.1, 0.9): 5}),
                       'max_depth': HyperWeights({None: 1, HyperRangeInt(2, 50): 2}),
                       'min_samples_split': HyperWeights({2: 2, HyperRangeFloat(0.1, 0.9): 1}),
                       'min_samples_leaf': HyperWeights({1: 2, HyperRangeFloat(0.01, 0.5): 1}),
                       'min_weight_fraction_leaf': HyperWeights({0: 2, HyperRangeFloat(0.01, 0.5): 1}),
                       'max_leaf_nodes': HyperWeights({None: 2, HyperRangeInt(2, 100): 1}),
                       'min_impurity_decrease': HyperWeights({1e-7: 2, HyperRangeFloat(0.001, 0.9): 1}),

                       'verbose': 0,
                       'random_state': 0,
                       'warm_start': False,
                       }

# parameters for extra trees
default_extra_trees = {**default_sklearn_trees, **{'n_jobs': -1}}
space_extra_trees = {**space_sklearn_trees, **{'n_jobs': -1}}

# TODO: add class weight for classifiers (extra trees, random forest)
space_extra_trees_classifier = {**space_extra_trees,
                                **{'criterion': HyperWeights({'gini': 1, 'entropy': 1}), 'class_weight': None, }
                                }

space_extra_trees_regressor = {**space_sklearn_trees, **{'criterion': HyperWeights({'mse': 1, 'mae': 1})}}

# parameters for random forest
default_random_forest = {**default_sklearn_trees, **{'n_jobs': -1}}

space_random_forest = {**space_sklearn_trees, **{'n_jobs': -1}}
space_random_forest_classifier = {**space_random_forest,
                                  **{'criterion': HyperWeights({'gini': 1, 'entropy': 1}), 'class_weight': None, }
                                  }

space_random_forest_regressor = {**space_random_forest,
                                 **{'criterion': HyperWeights({'mse': 1, 'mae': 1})
                                    }
                                 }

# parameters for adaboost
default_adaboost = {}
space_adaboost = {'n_estimators': HyperRangeInt(10, 100),
                  'learning_rate': HyperRangeFloat(0.001, 10.),
                  'random_state': 0,
                  }

space_adaboost_regressor = {**space_adaboost,
                            **{'loss': HyperWeights({'linear': 1, 'square': 1, 'exponential': 1})
                               }
                            }

space_adaboost_classifier = {**space_adaboost,
                             **{'algorithm': HyperWeights({'SAMME': 1, 'SAMME.R': 1})
                                }
                             }

# parameters for gradient boosting
default_gradient_boosting = {}
space_gradient_boosting_regressor = {**space_sklearn_trees,
                                     **{'learning_rate': HyperWeights({0.1: 2, HyperRangeFloat(0.001, 0.5): 1}),
                                        'loss': HyperWeights({'ls': 1, 'lad': 1, 'huber': 1, 'quantile': 1}),
                                        }}

space_gradient_boosting_classifier = {**space_sklearn_trees,
                                      **{'learning_rate': HyperWeights({0.1: 2, HyperRangeFloat(0.001, 0.5): 1}),
                                         'criterion': HyperWeights({'friedman_mse': 2, 'mse': 1, 'mae': 1}),
                                         'loss': HyperWeights({'deviance': 1, 'exponential': 1}),
                                         }}

# parameters for Logistic regression
default_logistic_regression = {'penalty': 'l2', 'dual': False, 'solver': 'liblinear', 'n_jobs': -1}
space_logistic_regression = {'penalty': HyperWeights({'l2': 1, 'l1': 1}),
                             'dual': HyperWeights({False: 1, True: 1}),
                             'tol': HyperWeights({1e-4: 2, HyperRangeFloat(0.0001, 0.01): 1}),
                             'C': HyperWeights({1.: 1, HyperRangeFloat(0.001, 0.9): 1}),
                             'fit_intercept': HyperWeights({True: 1, False: 1}),
                             'intercept_scaling': HyperWeights({1: 1, HyperRangeFloat(0.001, 0.9): 1}),
                             'solver': HyperWeights({'newton-cg': 1, 'liblinear': 1, 'lbfgs': 1, 'sag': 1, 'saga': 1}),
                             'max_iter': HyperWeights({1000: 1, HyperRangeInt(10, 1000): 1}),
                             'multi_class': HyperWeights({'ovr': 1, 'multinomial': 1}),
                             'n_jobs': -1
                             }

# parameters for Linear regression
default_linear_regression = {'n_jobs': -1}
space_linear_regression = {'fit_intercept': HyperWeights({True: 1, False: 1}),
                           'normalize': HyperWeights({False: 1, True: 1}), 'copy_X': False, 'n_jobs': -1
                           }

default_ridge_regression = {}
space_ridge_regression = {'alpha': HyperRangeFloat(0.001, 100.),
                          'fit_intercept': HyperWeights({True: 2, False: 1}),
                          'normalize': HyperWeights({False: 1, True: 2}),
                          'copy_X': False,
                          'tol': HyperWeights({1e-4: 2, HyperRangeFloat(0.0001, 0.01): 1}),
                          'solver': HyperWeights(
                              {'auto': 1, 'svd': 1, 'cholesky': 1, 'sparse_cg': 1, 'sag': 1, 'saga': 1}),
                          }

default_lasso_regression = {}
space_lasso_regression = {'alpha': HyperRangeFloat(0.001, 100.),
                          'fit_intercept': HyperWeights({True: 2, False: 1}),
                          'normalize': HyperWeights({False: 1, True: 2}),
                          'precompute': HyperWeights({False: 2, True: 1}),
                          'copy_X': False,
                          'tol': HyperWeights({1e-4: 2, HyperRangeFloat(0.0001, 0.01): 1}),
                          'positive': HyperWeights({False: 2, True: 1}),
                          'selection': HyperWeights({'cyclic': 2, 'random': 1}),
                          }

default_huber_regression = {}
space_huber_regression = {'epsilon': HyperRangeFloat(1.1, 10.),
                          'alpha': HyperRangeFloat(0.001, 100.),
                          'fit_intercept': HyperWeights({True: 2, False: 1}),
                          'tol': HyperWeights({1e-4: 2, HyperRangeFloat(0.0001, 0.01): 1}),
                          }

# parameters for Support Vector Machines (SVM)
default_linear_svc = {}
space_linear_svc = {'penalty': HyperWeights({'l1': 1, 'l2': 1}),
                    'loss': HyperWeights({'squared_hinge': 2, 'hinge': 1}),
                    'dual': HyperWeights({False: 1, True: 1}),
                    'tol': HyperWeights({1e-4: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'C': HyperWeights({1.: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'multi_class': HyperWeights({'ovr': 1, 'crammer_singer': 1}),
                    'fit_intercept': HyperWeights({True: 1, False: 1}),
                    'intercept_scaling': HyperWeights({1: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'max_iter': HyperWeights({1000: 1, HyperRangeInt(10, 1000): 1}),
                    }

default_linear_svr = {}
space_linear_svr = {'C': HyperWeights({1.: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'loss': HyperWeights({'epsilon_insensitive': 2, 'squared_epsilon_insensitive': 1}),
                    'epsilon': HyperWeights({0.1: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'dual': HyperWeights({True: 1, False: 1}),
                    'tol': HyperWeights({1e-4: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'fit_intercept': HyperWeights({True: 1, False: 1}),
                    'intercept_scaling': HyperWeights({1: 1, HyperRangeFloat(0.001, 0.9): 1}),
                    'max_iter': HyperWeights({1000: 1, HyperRangeInt(10, 1000): 1}),
                    'verbose': False,
                    }

default_svc = {'probability': True}
space_svc = {'C': HyperWeights({1.: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'kernel': HyperWeights({'rbf': 2, 'linear': 1, 'poly': 1, 'sigmoid': 1}),
             'degree': HyperWeights({3: 1, HyperRangeInt(2, 10): 1}),
             'gamma': HyperWeights({'auto': 1, HyperRangeFloat(0.001, 0.9): 1}),
             'coef0': HyperWeights({0: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'shrinking': HyperWeights({True: 1, False: 1}),
             'tol': HyperWeights({1e-3: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'max_iter': HyperWeights({-1: 1, HyperRangeInt(10, 1000): 1}),
             'verbose': False,
             'probability': True,
             }

default_svr = {}
space_svr = {'C': HyperWeights({1.: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'epsilon': HyperWeights({0.1: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'kernel': HyperWeights({'rbf': 2, 'linear': 1, 'poly': 1, 'sigmoid': 1}),
             'degree': HyperWeights({3: 1, HyperRangeInt(2, 10): 1}),
             'gamma': HyperWeights({'auto': 1, HyperRangeFloat(0.001, 0.9): 1}),
             'coef0': HyperWeights({0: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'shrinking': HyperWeights({True: 1, False: 1}),
             'tol': HyperWeights({1e-3: 1, HyperRangeFloat(0.001, 0.9): 1}),
             'max_iter': HyperWeights({-1: 1, HyperRangeInt(10, 1000): 1}),
             'verbose': False,
             }

# parameters for Naive Bayes
default_nb_multi = {'alpha': 1., 'fit_prior': True}
space_nb_multi = {'alpha': HyperRangeFloat(0.001, 10.),
                  'fit_prior': HyperWeights({True: 1, False: 1}),
                  }

default_nb_bernoulli = {'alpha': 1., 'binarize': True, 'fit_prior': True}
space_nb_bernoulli = {'alpha': HyperRangeFloat(0.001, 10.),
                      'binarize': HyperWeights({True: 1, False: 1}),
                      'fit_prior': HyperWeights({True: 1, False: 1}),
                      }

# parameters for KNN
default_knn = {'n_jobs': -1}
space_knn = {'n_neighbors': HyperWeights({5: 1, HyperRangeInt(2, 50): 1}),
             'weights': HyperWeights({'uniform': 2, 'distance': 1}),
             'algorithm': HyperWeights({'auto': 2, 'ball_tree': 1, 'kd_tree': 1, 'brute': 1}),
             'leaf_size': HyperWeights({30: 1, HyperRangeInt(2, 100): 1}),
             'p': HyperWeights({2: 1, HyperRangeInt(3, 10): 1}),
             'n_jobs': -1,
             }

# generic space for LightGBM (Sklearn API)
space_lightgbm = {'boosting_type': HyperWeights({'gbdt': 2, 'dart': 1}),
                  'num_leaves': HyperWeights({31: 1, HyperRangeInt(5, 200): 1}),
                  'max_depth': HyperWeights({-1: 1, HyperRangeInt(5, 100): 1}),
                  'learning_rate': HyperWeights({0.1: 1, HyperRangeFloat(0.001, 0.5): 1}),
                  'n_estimators': 5000,
                  'min_split_gain': HyperWeights({0: 1, HyperRangeFloat(0.0001, 0.01): 1}),
                  'min_child_weight': HyperWeights({1e-3: 1, HyperRangeFloat(0.001, 0.5): 1}),
                  'min_child_samples': HyperWeights({20: 1, HyperRangeInt(5, 100): 1}),
                  'subsample': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),
                  'subsample_freq': HyperWeights({0: 1, HyperRangeInt(1, 20): 1}),
                  'colsample_bytree': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),
                  'reg_alpha': HyperWeights({0: 1, HyperRangeFloat(0.001, 0.5): 1}),
                  'reg_lambda': HyperWeights({0: 1, HyperRangeFloat(0.001, 0.5): 1}),
                  'verbose': -1
                  }

default_lightgbm_regressor = {'objective': 'regression', 'metric': 'mse', 'verbose': 0}
space_lightgbm_regressor = {**space_lightgbm,
                            **{'objective': HyperWeights(
                                {'regression': 1, 'regression_l1': 1, 'huber': 1, 'fair': 1, 'poisson': 1}),
                                'metric': 'mse'
                            }
                            }

default_lightgbm_classifier = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': 0}
space_lightgbm_classifier = {**space_lightgbm,
                             **{'objective': 'binary',
                                'metric': 'binary_logloss',
                                }}

# parameters for Xgboost (sklearn API)
space_xgboost = {'max_depth': HyperRangeInt(2, 100),
                 'learning_rate': HyperWeights({0.3: 1, HyperRangeFloat(0.001, 0.5): 1}),
                 'n_estimators': 5000,
                 'booster': HyperWeights({'gbtree': 2, 'gblinear': 1, 'dart': 100}),

                 'gamma': HyperWeights({0: 1, HyperRangeFloat(0.01, 100.): 1}),
                 'min_child_weight': HyperWeights({1: 1, HyperRangeFloat(0.01, 1000.): 1}),
                 'max_delta_step': HyperWeights({0: 1, HyperRangeInt(1, 10): 1}),
                 'subsample': HyperWeights({1.: 1, HyperRangeFloat(0.01, 0.99): 1}),
                 'colsample_bytree': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),
                 'colsample_bylevel': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),

                 'reg_alpha': HyperWeights({0: 1, HyperRangeFloat(0.01, 0.99): 1}),
                 'reg_lambda': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),
                 'scale_pos_weight': HyperWeights({1: 1, HyperRangeFloat(0.01, 0.99): 1}),

                 'tree_method': HyperWeights({'auto': 1, 'exact': 1, 'approx': 1}),
                 'sketch_eps': HyperWeights({0.03: 1, HyperRangeFloat(0.01, 0.99): 1}),
                 'n_jobs': -1,
                 'silent': 1
                 }

default_xgboost_classifier = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'silent': 1}
space_xgboost_classifier = {**space_xgboost,
                            **{  # 'booster': HyperWeights({'gbtree': 2}),
                                'objective': 'binary:logistic',
                                'eval_metric': 'logloss',
                            }}

default_xgboost_regressor = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1}
space_xgboost_regressor = {**space_xgboost,
                           **{  # 'booster': HyperWeights({'gbtree': 2}),
                               'objective': 'reg:linear',
                               'eval_metric': 'rmse',
                           }}

# parameters for  Catboost
# TODO: extend parameters
space_catboost = {'learning_rate': HyperWeights({0.3: 1, HyperRangeFloat(0.001, 0.5): 1}),
                  'depth': HyperWeights({4: 1, HyperRangeInt(1, 10): 1}),
                  'verbose': False
                  }

default_catboost_classifier = {}
space_catboost_classifier = space_catboost
default_catboost_regressor = {}
space_catboost_regressor = space_catboost

# parameters for keras model (Neural networks)

default_keras = {'units': 64, 'batch_size': 128, 'batch_normalization': False, 'activation': 'relu',
                 'optimizer': 'RMSprop', 'learning_rate': 0.01, 'number_layers': 2, 'dropout': 0.5}

space_keras = {'units': HyperWeights({16: 1, 32: 1, 64: 1, 128: 2, 256: 5, 512: 5, 1024: 2, 2048: 1}),
               'batch_size': HyperWeights({16: 1, 32: 1, 64: 1, 128: 5, 256: 10, 512: 5, 1024: 2, 2048: 1}),
               'batch_normalization': HyperWeights({True: 1, False: 1}),
               'activation': HyperWeights(
                   {'relu': 1, 'tanh': 1, 'sigmoid': 1, 'hard_sigmoid': 1, 'linear': 1}),
               'optimizer': HyperWeights({'RMSprop': 2, 'Adagrad': 1, 'Adadelta': 1, 'Adam': 1}),
               'learning_rate': HyperRangeFloat(0.001, 0.5),
               'number_layers': HyperWeights({1: 1, 2: 2, 3: 5, 4: 2, 5: 2, 6: 2, 7: 1}),
               'dropout': HyperRangeFloat(0.1, 0.8),
               }

# parameters for ensemble selection
default_ensemble = {'rounds': 20}
space_ensemble = {'rounds': HyperRangeInt(5, 200)}
