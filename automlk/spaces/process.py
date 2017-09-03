from .hyper import *

"""
this module defines the hyper-parameter spaces for the various preprocssing steps 
"""

# categorical encoding

default_categorical = {'encoder': 'BaseNEncoder', 'drop_invariant': True}

space_categorical = {'encoder': HyperWeights({'BaseNEncoder': 1,
                                              # 'BackwardDifferenceEncoder': 1,
                                              # 'BinaryEncoder': 1,
                                              'HashingEncoder': 1,
                                              'OneHotEncoder': 1,
                                              # 'SumEncoder': 1,
                                              # 'PolynomialEncoder': 1
                                              }),

                     'drop_invariant': HyperWeights({True: 1, False: 1})
                     }

# missing values imputing

default_missing = {'strategy': 'fixed_0'}

space_missing = {'strategy': HyperWeights({'fixed_0': 1,
                                           'fixed_m1': 1,
                                           'mean': 1,
                                           'median': 1,
                                           'most_frequent': 1,
                                           })
                 }

# scaling
default_scaling = {'scaler': 'standard'}

space_scaling = {'scaler': HyperWeights({'standard': 1,
                                         'min_max': 1,
                                         'robust': 1,
                                         'max_abs': 1,
                                         })
                 }
