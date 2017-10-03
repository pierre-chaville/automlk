from .preprocessing import *


class PpSolution(object):
    # to define a pre-processing and the parameters / conditions of usage

    def __init__(self, ref, name, process, default_params, space_params, pp_type, selectable=True, limit_size=1e32):
        self.ref = ref
        self.name = name
        self.process = process
        self.default_params = default_params
        self.space_params = space_params
        self.pp_type = pp_type
        self.selectable = selectable
        self.limit_size = limit_size


# list of solutions
pp_solutions = [
    # data processing
    PpSolution('CE', 'Categorical encoding', HyperProcessCategorical, default_categorical, space_categorical, 'data'),
    PpSolution('MISS', 'Missing values', HyperProcessMissing, default_missing, space_missing, 'data'),
    PpSolution('SCALE', 'Feature Scaling', HyperProcessScaling, default_scaling, space_scaling, 'data'),

    # feature processing
    # PpSolution('SVD', 'Truncated SVD', HyperProcessTruncatedSVD, default_truncated_svd, space_truncated_svd, 'feature'),
    # PpSolution('ICA', 'Fast ICA', HyperProcessFastICA, default_fast_ica, space_fast_ica, 'feature', limit_size=1000),
    PpSolution('NULL', 'PassThrough', HyperProcessPassThrough, default_passthrough, space_passthrough, 'feature'),
]

# mapping table
pp_solutions_map = {s.ref: s for s in pp_solutions}