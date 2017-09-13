from abc import ABCMeta, abstractmethod
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Imputer, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD, FastICA
from .spaces.process import *


def pre_processing(context, X_train, y_train, X_test, y_test):
    # performs the different pre-processing steps

    process_list = []

    # X pre-processing: categorical
    if len(context.cat_cols) > 0:
        process_list.append(HyperProcessCategorical)

    # missing values
    if len(context.missing_cols) > 0:
        process_list.append(HyperProcessMissing)

    # scaling
    process_list.append(HyperProcessScaling)

    # choice between feature transformations
    choices = [HyperProcessPassThrough, HyperProcessTruncatedSVD]
    process_features = random.choices(choices, weights=[5, 1])[0]
    process_list.append(process_features)

    # then execute pipeline
    context, X_train, y_train, X_test, y_test = execute_pipeline(context, process_list, X_train.copy(), y_train.copy(),
                                                                 X_test.copy(), y_test.copy())


    return context, X_train, y_train, X_test, y_test


def execute_pipeline(context, process_list, X_train, y_train, X_test, y_test):
    pipeline = []
    for p_class in process_list:
        process = p_class(context)
        process.set_random_params()
        print('executing process', process.process_name, process.params)
        X_train = process.fit_transform(X_train, y_train)
        X_test = process.transform(X_test)
        context.pipeline.append(process)
    return context, X_train, y_train, X_test, y_test


class HyperProcess(object):
    __metaclass__ = ABCMeta

    # abstract class for model preprocessing in hyper optimisation

    @abstractmethod
    def __init__(self, context):
        self.context = context
        self.transformer = None
        self.params = {}

    @abstractmethod
    def set_default_params(self):
        pass

    @abstractmethod
    def set_random_params(self):
        pass

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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'Categorical encoding'

    def set_default_params(self):
        self.params = default_categorical

    def set_random_params(self):
        self.params = get_random_params(space_categorical)

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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'Missing values'

    def set_default_params(self):
        self.params = default_missing

    def set_random_params(self):
        self.params = get_random_params(space_missing)

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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'Feature Scaling'

    def set_default_params(self):
        self.params = default_scaling

    def set_random_params(self):
        self.params = get_random_params(space_scaling)

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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'Truncated SVD'

    def set_default_params(self):
        self.params = default_truncated_svd
        self.__check_params()

    def set_random_params(self):
        self.params = get_random_params(space_truncated_svd)
        self.__check_params()

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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'Fast ICA'

    def set_default_params(self):
        self.params = default_fast_ica
        self.__check_params()

    def set_random_params(self):
        self.params = get_random_params(space_fast_ica)
        self.__check_params()

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

    def __init__(self, context):
        super().__init__(context)
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

    def __init__(self, context):
        super().__init__(context)
        self.process_name = 'PassThrough'

    def transform(self, X):
        return X


