from abc import ABCMeta, abstractmethod
import pandas as pd


class HyperProcess(object):
    __metaclass__ = ABCMeta

    # abstract class for model preprocessing in hyper optimisation

    @abstractmethod
    def __init__(self, **params):
        print(params)
        self.set_params(**params)
        self.transformer = None
        self.__feature_names = []

    @abstractmethod
    def get_params(self, deep=True):
        return self.params

    @abstractmethod
    def set_params(self, **params):
        self.params = params
        if 'context' in params.keys():
            self.context = params['context']
        else:
            self.context = {}
        self.t_params = {key:params[key] for key in params.keys() if key not in ['context']}

    @abstractmethod
    def get_feature_names(self):
        return self.__feature_names

    @abstractmethod
    def fit(self, X, y):
        # fit the transformer with the data
        if isinstance(X, pd.DataFrame):
            self.__feature_names = X.columns

    @abstractmethod
    def transform(self, X):
        # fit and transform
        Xt = self.transformer.transform(X)
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
            Xt.columns = self.__feature_names
        return Xt

    @abstractmethod
    def fit_transform(self, X, y):
        # fit and transform
        self.fit(X, y)
        return self.transform(X)


class HyperProcessLabel(HyperProcess):
    # class for process categorical encoding - label encoder

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = []

    def fit(self, X, y):
        self.transformer = []
        for col in self.context.cat_cols:
            encoder = {x: i for i, x in enumerate(X[col].unique())}
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            X[col] = X[col].map(lambda x: encoder[x] if x in encoder else -1)
        return X

p1 = HyperProcess(a=234)
p2 = HyperProcessLabel(a=567)