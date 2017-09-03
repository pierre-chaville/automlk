from abc import ABCMeta, abstractmethod
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Imputer
from .spaces.hyper import get_random_params
from .spaces.process import *


class HyperProcess(object):
    __metaclass__ = ABCMeta

    # abstract class for model preprocessing in hyper optimisation

    @abstractmethod
    def __init__(self, context):
        self.context = context
        self.transformer = None
        self.params = {}

    @abstractmethod
    def fit(self, X, y):
        # fit the transformer with the data
        pass

    @abstractmethod
    def transform(self, X):
        # fit and transform
        return X

    @abstractmethod
    def fit_transform(self, X, y):
        # fit and transform
        self.fit(X, y)
        return self.transform(X)


class HyperProcessCategorical(HyperProcess):
    # class for process categorical encoding

    def __init__(self, dataset):
        super().__init__(dataset)
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

    def transform(self, X):
        return self.transformer.transform(X)


class HyperProcessMissing(HyperProcess):
    # class for transformation of missing values

    def __init__(self, dataset):
        super().__init__(dataset)
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

    def __init__(self, dataset):
        super().__init__(dataset)
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

    def transform(self, X):
        return self.transformer.transform(X)


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
    context, X_train, y_train, X_test, y_test = execute_pipeline(context, process_list, X_train.copy(), y_train.copy(),
                                                                 X_test.copy(), y_test.copy())

    return context, X_train, y_train, X_test, y_test
