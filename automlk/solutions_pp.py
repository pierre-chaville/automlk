from .preprocessing import *


class PpSolution(object):
    # to define a pre-processing and the parameters / conditions of usage

    def __init__(self, ref, name, process, default_params, space_params, pp_type, selectable=True, limit_size=1e32,
                 default_solution=False):
        self.ref = ref
        self.name = name
        self.process = process
        self.default_params = default_params
        self.space_params = space_params
        self.pp_type = pp_type
        self.selectable = selectable
        self.limit_size = limit_size
        self.default_solution = default_solution


# list of solutions
pp_solutions = [
    # solutions for categorical encoding
    PpSolution('CE-HOT', 'One hot categorical', HyperProcessOneHot, default_categorical, space_categorical,
               'categorical', default_solution=True),
    PpSolution('CE-BASE', 'BaseN categorical', HyperProcessBaseN, default_categorical, space_baseN, 'categorical'),
    PpSolution('CE-HASH', 'Hashing categorical', HyperProcessHashing, default_categorical, space_categorical,
               'categorical'),

    # solutions for missing values
    PpSolution('MISS-FIXED', 'Missing values fixed', HyperProcessMissingFixed, default_missing_fixed,
               space_missing_fixed, 'missing', default_solution=True),
    PpSolution('MISS', 'Missing values', HyperProcessMissing, default_missing, space_missing, 'missing'),

    # solutions for text processing
    PpSolution('BOW', 'Bag of words', HyperProcessBOW, default_bow, space_bow, 'text'),
    PpSolution('W2V', 'Word2Vec', HyperProcessWord2Vec, default_word2vec, space_word2vec, 'text', default_solution=True),

    # scaling solutions
    PpSolution('SCALE', 'Feature Scaling', HyperProcessScaling, default_scaling, space_scaling, 'scaling'),
    PpSolution('NO-SCALE', 'No Scaling', HyperProcessNoScaling, NO_PARAMS, NO_PARAMS, 'scaling', default_solution=True),

    # feature processing (selection, reduction)
    PpSolution('PASS', 'No Feature selection', HyperProcessPassThrough, NO_PARAMS, NO_PARAMS, 'feature',
               default_solution=True),
    PpSolution('SVD', 'Truncated SVD', HyperProcessTruncatedSVD, default_truncated_svd, space_truncated_svd, 'feature', limit_size=50),
    PpSolution('ICA', 'Fast ICA', HyperProcessFastICA, default_fast_ica, space_fast_ica, 'feature', limit_size=50),
    PpSolution('PCA', 'PCA', HyperProcessPCA, default_pca, space_pca, 'feature', limit_size=50),
    PpSolution('FS-RF', 'Selection RF', HyperProcessSelectionRf, default_sel_rf, space_sel_rf, 'feature'),
]

# mapping table
pp_solutions_map = {s.ref: s for s in pp_solutions}
