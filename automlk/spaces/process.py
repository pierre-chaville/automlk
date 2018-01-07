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
             'first_words': HyperWeights({0: 5, 3: 1, 5: 1, 10: 1})
             }

# word2vec
default_word2vec = {'size': 100, 'iter': 20, 'window': 5, 'min_count': 5, 'sg': 0, 'workers': 8}
space_word2vec = {'size': HyperChoice([50, 100, 200, 400]),
                  'iter': HyperChoice([10, 20, 50]),
                  'window': HyperChoice([5, 7, 11]),
                  'min_count': HyperChoice([2, 5, 10]),
                  'sg': HyperChoice([0, 1]),
                  'workers': 8
                  }

# fasttext
default_fasttext = {'size': 100, 'iter': 1, 'window': 5, 'min_count': 5, 'sg': 0, 'workers': 8}
space_fasttext = {'size': HyperChoice([50, 100, 200]),
                  'iter': HyperChoice([1, 2, 5]),
                  'window': HyperChoice([5, 7, 11]),
                  'min_count': HyperChoice([2, 5, 10]),
                  'sg': HyperChoice([0, 1]),
                  'workers': 8
                  }

# doc2vec
default_doc2vec = {'size': 100, 'iter': 50, 'window': 5, 'min_count': 5, 'dm': 0, 'workers': 8}

space_doc2vec = {'size': HyperChoice([50, 100, 200, 400]),
                 'iter': HyperChoice([20, 50, 100]),
                 'window': HyperChoice([5, 7, 11]),
                 'min_count': HyperChoice([2, 5, 10]),
                 'dm': HyperChoice([0, 1]),
                 'workers': 8
                 }

# list of bow configurations for unsupervised text sets
space_textset_bow = [{'max_features': 200, 'ngram_range': (1, 1), 'tfidf': True},
                     {'max_features': 200, 'ngram_range': (1, 1), 'tfidf': False},
                     {'max_features': 500, 'ngram_range': (1, 1), 'tfidf': True},
                     {'max_features': 500, 'ngram_range': (1, 1), 'tfidf': False},
                     {'max_features': 500, 'ngram_range': (1, 2), 'tfidf': True},
                     {'max_features': 600, 'ngram_range': (1, 2), 'tfidf': False},
                     {'max_features': 1000, 'ngram_range': (1, 1), 'tfidf': True},
                     {'max_features': 1000, 'ngram_range': (1, 1), 'tfidf': False},
                     {'max_features': 1000, 'ngram_range': (1, 2), 'tfidf': True},
                     {'max_features': 1000, 'ngram_range': (1, 2), 'tfidf': False},
                     {'max_features': 2000, 'ngram_range': (1, 3), 'tfidf': True},
                     {'max_features': 2000, 'ngram_range': (1, 3), 'tfidf': False},
                     ]

default_textset_bow = space_textset_bow[0]

# list of word2vec configurations for unsupervised text sets
space_textset_w2v = [{'size': 100, 'iter': 10, 'window': 5, 'min_count': 2, 'sg': 0, 'workers': 8},
                     {'size': 100, 'iter': 20, 'window': 7, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 100, 'iter': 50, 'window': 11, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 5, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 7, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 11, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 400, 'iter': 20, 'window': 5, 'min_count': 5, 'sg': 0, 'workers': 8},
                     {'size': 400, 'iter': 50, 'window': 11, 'min_count': 5, 'sg': 0, 'workers': 8}]

default_textset_w2v = space_textset_w2v[0]

# list of doc2vec configurations for unsupervised text sets
space_textset_d2v = [{'size': 100, 'iter': 10, 'window': 5, 'min_count': 2, 'dm': 0, 'workers': 8},
                     {'size': 100, 'iter': 20, 'window': 7, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 100, 'iter': 50, 'window': 11, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 5, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 7, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 200, 'iter': 20, 'window': 11, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 400, 'iter': 20, 'window': 5, 'min_count': 5, 'dm': 0, 'workers': 8},
                     {'size': 400, 'iter': 50, 'window': 11, 'min_count': 5, 'dm': 0, 'workers': 8}]

default_textset_d2v = space_textset_d2v[0]

# missing values imputing

default_missing_fixed = {'fixed': 0}
space_missing_fixed = {'fixed': HyperChoice([-1000, -100, -1, 0])
                       }

default_missing_frequency = {'frequency': 0.01}
space_missing_frequency = {'frequency': HyperChoice([0.1, 0.01, 0.001])
                           }

# scaling
default_scaling_robust = {'quantile_range': (0.01, 0.99)}
space_scaling_robust = {'quantile_range': HyperWeights({(0.001, 0.999): 5,
                                                        (0.01, 0.99): 10,
                                                        (0.1, 0.9): 5,
                                                        (0.25, 0.75): 1})
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
