from abc import ABCMeta, abstractmethod
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Imputer, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD, FastICA
from .spaces.process import *


class HyperProcess(object):
    __metaclass__ = ABCMeta

    # abstract class for model preprocessing in hyper optimisation

    @abstractmethod
    def __init__(self, context, params):
        self.context = context
        self.params = params
        self.transformer = None

    @abstractmethod
    def fit(self, X, y):
        # fit the transformer with the data
        pass

    @abstractmethod
    def transform(self, X):
        # fit and transform
        return self.transformer.transform(X)

    @abstractmethod
    def fit_transform(self, X, y):
        # fit and transform
        self.fit(X, y)
        return self.transform(X)


class HyperProcessCategorical(HyperProcess):
    # class for process categorical encoding

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        encoder = getattr(ce, self.params['encoder'])
        self.transformer = encoder(cols=self.context.cat_cols, drop_invariant=self.params['drop_invariant'])
        self.transformer.fit(X, y)

        # update new list of columns
        Xt = self.transformer.transform(X.copy())
        self.context.cat_cols = []
        self.context.feature_names = Xt.columns
        print('the context has now %d features' % len(self.context.feature_names))


class HyperProcessMissing(HyperProcess):
    # class for transformation of missing values

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        if self.params['strategy'] in ['mean', 'median', 'most_frequent']:
            self.transformer = Imputer(strategy=self.params['strategy'])
            self.transformer.fit(X, y)

    def transform(self, X):
        if self.params['strategy'] in ['mean', 'median', 'most_frequent']:
            return self.transformer.transform(X)
        elif self.params['strategy'] == 'fixed_0':
            return X.fillna(0)
        elif self.params['strategy'] == 'fixed_m1':
            return X.fillna(-1)


class HyperProcessScaling(HyperProcess):
    # class for scaling transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        if self.params['scaler'] == 'standard':
            self.transformer = StandardScaler()
        elif self.params['scaler'] == 'min_max':
            self.transformer = MinMaxScaler()
        elif self.params['scaler'] == 'max_abs':
            self.transformer = MaxAbsScaler()
        elif self.params['scaler'] == 'robust':
            self.transformer = RobustScaler()

        self.transformer.fit(X, y)



class HyperProcessTruncatedSVD(HyperProcess):
    # class for Truncated SVD feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def __check_params(self):
        self.params['n_components'] = int(self.params['reduction_ratio'] * len(self.context.feature_names))
        self.params.pop('reduction_ratio', None)

    def fit(self, X, y):
        self.transformer = TruncatedSVD(**self.params)
        self.transformer.fit(X)
        self.context.feature_names = ['SVD_%d' for i in range(self.params['n_components'])]
        print('the context has now %d features' % len(self.context.feature_names))


class HyperProcessFastICA(HyperProcess):
    # class for Fast ICA feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def __check_params(self):
        self.params['n_components'] = int(self.params['reduction_ratio'] * len(self.context.feature_names))
        self.params.pop('reduction_ratio', None)

    def fit(self, X, y):
        self.transformer = FastICA(**self.params)
        self.transformer.fit(X)
        self.context.feature_names = ['ICA_%d' for i in range(self.params['n_components'])]
        print('the context has now %d features' % len(self.context.feature_names))


class HyperProcessPolynomial(HyperProcess):
    # class for Polynomial feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)
        self.process_name = 'Polynomial'

    def set_default_params(self):
        self.params = default_polynomial

    def set_random_params(self):
        self.params = get_random_params(space_polynomial)

    def fit(self, X, y):
        self.transformer = PolynomialFeatures(**self.params)
        self.transformer.fit(X)

        self.context.feature_names = self.transformer.get_feature_names(self.context.feature_names)
        print('the context has now %d features' % len(self.context.feature_names))


class HyperProcessPassThrough(HyperProcess):
    # class for No transformation

    def __init__(self, context, params):
        super().__init__(context, params)
        self.process_name = 'PassThrough'

    def transform(self, X):
        return X


