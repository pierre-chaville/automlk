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

# truncated SVD

default_truncated_svd = {'reduction_ratio': 0.5}

space_truncated_svd = {'reduction_ratio': HyperRangeFloat(0.2, 0.6),
                       #'algorithm': HyperWeights({'randomized': 1, 'arpack': 1}),
                       'algorithm': 'arpack',
                       'n_iter': HyperRangeInt(3, 6),
                       'tol': HyperWeights({0.001: 1, 0.0001: 1, 0.00001: 1})
                       }

# fast ICA

default_fast_ica = {'reduction_ratio': 0.5}

space_fast_ica = {'reduction_ratio': HyperRangeFloat(0.1, 0.9),
                  'algorithm': HyperWeights({'parallel': 2, 'deflation': 1}),
                  'max_iter': HyperRangeInt(50, 200),
                  'tol': HyperWeights({0: 2, 0.001: 1, 0.0001: 1, 0.00001: 1})
                  }

# fast ICA

default_polynomial = {'degree': 2}

space_polynomial = {'degree': 2, #HyperRangeInt(1, 2),
                    'interaction_only': True, #HyperWeights({False: 2, False: 1}),
                    'include_bias': False #HyperWeights({True: 2, False: 1}),
                    }

# Pass Through
default_passthrough = {}
space_passthrough = {}