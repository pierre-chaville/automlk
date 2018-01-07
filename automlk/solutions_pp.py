from .preprocessing import *
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

PP_CATEGORIES = ['categorical', 'missing', 'text', 'scaling', 'feature', 'sampling']


class PpSolution(object):
    # to define a pre-processing and the parameters / conditions of usage

    def __init__(self, ref, name, process, default_params, space_params, pp_type, selectable=True, limit_size=1e32,
                 problem_type='*'):
        self.ref = ref
        self.name = name
        self.process = process
        self.default_params = default_params
        self.space_params = space_params
        self.pp_type = pp_type
        self.selectable = selectable
        self.limit_size = limit_size
        self.problem_type = problem_type


# list of solutions
pp_solutions = [
    # solutions for categorical encoding
    PpSolution('CE-PASS', 'No encoding', TransformerPassThrough, {}, {}, 'categorical'),
    PpSolution('CE-LAB', 'Label Encoder', TransformerLabel, {}, {}, 'categorical'),
    PpSolution('CE-HOT', 'One hot categorical', TransformerOneHot, default_categorical, space_categorical,
               'categorical'),
    PpSolution('CE-BASE', 'BaseN categorical', TransformerBaseN, default_categorical, space_baseN, 'categorical'),
    PpSolution('CE-HASH', 'Hashing categorical', TransformerHashing, default_categorical, space_categorical,
               'categorical'),

    # solutions for missing values
    PpSolution('MS-PASS', 'No missing', TransformerPassThrough, {}, {}, 'missing'),
    PpSolution('MS-FIXED', 'Missing values fixed', TransformerMissingFixed, default_missing_fixed,
               space_missing_fixed, 'missing'),
    PpSolution('MS-FREQ', 'Missing values frequencies', TransformerMissingFrequency, default_missing_frequency,
               space_missing_frequency, 'missing'),

    # solutions for text processing
    PpSolution('TX-BOW', 'Bag of words', TransformerBOW, default_textset_bow, space_textset_bow, 'text'),
    PpSolution('TX-W2V', 'Word2Vec', TransformerWord2Vec, default_textset_w2v, space_textset_w2v, 'text',
               selectable=import_gensim),
    PpSolution('TX-D2V', 'Doc2Vec', TransformerDoc2Vec, default_textset_d2v, space_textset_d2v, 'text',
               selectable=import_gensim),

    # scaling solutions
    PpSolution('SC-PASS', 'No scaling', TransformerPassThrough, {}, {}, 'scaling'),
    PpSolution('SC-STD', 'Scaling Standard', TransformerScalingStandard, {}, {}, 'scaling'),
    PpSolution('SC-MINMAX', 'Scaling MinMax', TransformerScalingMinMax, {}, {}, 'scaling'),
    PpSolution('SC-MAXABS', 'Scaling MaxAbs', TransformerScalingMaxAbs, {}, {}, 'scaling'),
    PpSolution('SC-ROBUST', 'Scaling Robust', TransformerScalingRobust,
               default_scaling_robust, space_scaling_robust, 'scaling'),

    # feature reduction
    PpSolution('FR-PASS', 'No Feature selection', TransformerPassThrough, {}, {}, 'feature'),
    PpSolution('FR-SVD', 'Truncated SVD', TransformerTruncatedSVD, default_truncated_svd, space_truncated_svd,
               'feature',
               limit_size=50),
    PpSolution('FR-ICA', 'Fast ICA', TransformerFastICA, default_fast_ica, space_fast_ica, 'feature', limit_size=50),
    PpSolution('FR-PCA', 'PCA', TransformerPCA, default_pca, space_pca, 'feature', limit_size=50),

    # feature selection from model
    PpSolution('FR-RFR', 'Selection RF', TransformerSelectionRfR, default_sel_rf, space_sel_rf, 'feature',
               problem_type='regression'),
    PpSolution('FR-RFC', 'Selection RF', TransformerSelectionRfC, default_sel_rf, space_sel_rf, 'feature',
               problem_type='classification'),
    PpSolution('FR-LR', 'Selection LSVR', TransformerSelectionLinearSVR, {}, {}, 'feature', problem_type='regression'),

    # sampling solutions
    PpSolution('SP-PASS', 'No re-sampling', NoSampling, {}, {}, 'sampling'),
    PpSolution('SP-ROS', 'Random Over', RandomOverSampler, {}, {}, 'sampling'),
    PpSolution('SP-SMOTE', 'SMOTE', SMOTE, {}, {}, 'sampling'),

]

# mapping table
pp_solutions_map = {s.ref: s for s in pp_solutions}

# default pre-processing lists
pp_def_lgbm = ['CE-PASS', 'MS-FIXED', 'TX-W2V', 'SC-PASS', 'FR-PASS']
pp_def_trees = ['CE-LAB', 'MS-FIXED', 'TX-W2V', 'SC-PASS', 'FR-PASS']
pp_def_knn = ['CE-HOT', 'MS-FIXED', 'TX-W2V', 'SC-STD', 'FR-PASS']
pp_def_linear = ['CE-HOT', 'MS-FIXED', 'TX-W2V', 'SC-ROBUST', 'FR-PASS']
pp_def_NN = ['CE-HOT', 'MS-FIXED', 'TX-W2V', 'SC-MINMAX', 'FR-PASS']

pp_list_lgbm = ['CE-LAB', 'CE-HOT', 'CE-BASE', 'CE-HASH',
                'MS-FIXED', 'MS-FREQ',
                'TX-BOW', 'TX-W2V', 'TX-D2V',
                'SC-PASS', 'SC-STD', 'SC-MINMAX', 'SC-MAXABS', 'SC-ROBUST',
                'FR-PASS', 'FR-SVD', 'FR-ICA', 'FR-PCA', 'FR-RFR', 'FR-RFC', 'FR-LR']

pp_list_trees = ['CE-LAB', 'CE-HOT', 'CE-BASE', 'CE-HASH',
                 'MS-FIXED', 'MS-FREQ',
                 'TX-BOW', 'TX-W2V', 'TX-D2V',
                 'SC-PASS', 'SC-STD', 'SC-MINMAX', 'SC-MAXABS', 'SC-ROBUST',
                 'FR-PASS', 'FR-SVD', 'FR-ICA', 'FR-PCA', 'FR-RFR', 'FR-RFC', 'FR-LR']

pp_list_knn = ['CE-HOT', 'CE-BASE', 'CE-HASH',
               'MS-FIXED', 'MS-FREQ',
               'TX-BOW', 'TX-W2V', 'TX-D2V',
               'SC-STD', 'SC-MINMAX', 'SC-MAXABS', 'SC-ROBUST',
               'FR-PASS', 'FR-SVD', 'FR-ICA', 'FR-PCA', 'FR-RFR', 'FR-RFC', 'FR-LR']

pp_list_linear = ['CE-HOT', 'CE-BASE', 'CE-HASH',
                  'MS-FIXED', 'MS-FREQ',
                  'TX-BOW', 'TX-W2V', 'TX-D2V',
                  'SC-STD', 'SC-MINMAX', 'SC-MAXABS', 'SC-ROBUST',
                  'FR-PASS', 'FR-SVD', 'FR-ICA', 'FR-PCA', 'FR-RFR', 'FR-RFC', 'FR-LR']

pp_list_NN = ['CE-HOT', 'CE-BASE', 'CE-HASH',
              'MS-FIXED', 'MS-FREQ',
              'TX-BOW', 'TX-W2V', 'TX-D2V',
              'SC-STD', 'SC-MINMAX', 'SC-MAXABS', 'SC-ROBUST',
              'FR-PASS', 'FR-SVD', 'FR-ICA', 'FR-PCA', 'FR-RFR', 'FR-RFC', 'FR-LR']
