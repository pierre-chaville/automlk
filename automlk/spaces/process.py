from .hyper import *

"""
this module defines the hyper-parameter spaces for the various pre-processing steps
"""

NO_PARAMS = {}

# categorical encoding

default_categorical = {'drop_invariant': True}
space_categorical = {'drop_invariant': HyperWeights({True: 1, False: 1})}
space_baseN = {'drop_invariant': HyperWeights({True: 1, False: 1}), 'base': HyperWeights({2: 1, 3: 1, 5: 1})}

# text encoding

# bag of words
default_bow = {'max_features': 500, 'ngram_range': (1, 2), 'tfidf': False, 'first_words': 0}
space_bow = {'max_features': HyperChoice([200, 500, 1000]),
             'ngram_range': HyperChoice([(1, 1), (1, 2), (1, 3)]),
             'tfidf': HyperWeights({True: 1, False: 1}),
             'first_words': HyperWeights({0: 1, 3: 1, 5: 1, 10: 1, 20: 1, 100: 1})
             }

# word2vec
default_word2vec = {'dim': 100, 'n_iter': 10}
space_word2vec = {'dim': HyperChoice([100, 200, 300]),
                  'n_iter': HyperChoice([5, 10, 20])
                  }

# missing values imputing

default_missing_fixed = {'fixed': 0}
space_missing_fixed = {'fixed': HyperChoice([-1000, -100, -1, 0])
                       }

default_missing = {'strategy': 'mean'}
space_missing = {'strategy': HyperChoice(['mean', 'median', 'most_frequent'])}

# scaling
default_scaling = {'scaler': 'standard'}
space_scaling = {'scaler': HyperWeights({'standard': 1,
                                         'min_max': 1,
                                         'robust': 1,
                                         'max_abs': 1,
                                         })
                 }

# truncated SVD

default_truncated_svd = {'n_components': 10}
space_truncated_svd = {'n_components': HyperChoice([10, 20, 50, 100]),
                       'algorithm': 'arpack',
                       }

# fast ICA

default_fast_ica = {'n_components': 10}

space_fast_ica = {'n_components': HyperChoice([10, 20, 50, 100]),
                  'algorithm': HyperWeights({'parallel': 2, 'deflation': 1}),
                  }

# PCA

default_pca = {'n_components': 10}

space_pca = {'n_components': HyperChoice([10, 20, 50, 100])}

# polynamial

default_polynomial = {'degree': 2}

space_polynomial = {'degree': 2,  # HyperRangeInt(1, 2),
                    'interaction_only': True,  # HyperWeights({False: 2, False: 1}),
                    'include_bias': False  # HyperWeights({True: 2, False: 1}),
                    }
# select from model - random forest
default_sel_rf = {'n_estimators': 20}

space_sel_rf = {'n_estimators': HyperChoice([20, 50])}
