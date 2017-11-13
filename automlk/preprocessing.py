from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Imputer, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD, FastICA, PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .spaces.process import *
from .utils.text_encoders import *

try:
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    import_gensim = True
except:
    import_gensim = False


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

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, context, params):
        super().__init__(context, params)

    @abstractmethod
    def fit(self, X, y):
        # encoder = getattr(ce, self.params['encoder'])
        # self.transformer = encoder(cols=self.context.cat_cols, drop_invariant=self.params['drop_invariant'])
        self.transformer.fit(X, y)
        # update new list of columns
        Xt = self.transformer.transform(X.copy())
        self.context.cat_cols = []
        self.context.feature_names = Xt.columns


class HyperProcessOneHot(HyperProcessCategorical):
    # class for process categorical encoding - one hot

    def __init__(self, context, params):
        super().__init__(context, params)
        self.transformer = ce.OneHotEncoder(**params)


class HyperProcessBaseN(HyperProcessCategorical):
    # class for process categorical encoding - base N

    def __init__(self, context, params):
        super().__init__(context, params)
        self.transformer = ce.BaseNEncoder(**params)


class HyperProcessHashing(HyperProcessCategorical):
    # class for process categorical encoding - hashing

    def __init__(self, context, params):
        super().__init__(context, params)
        self.transformer = ce.HashingEncoder(**params)


class HyperProcessBOW(HyperProcess):
    # class for process bag of words for text

    def __init__(self, context, params):
        super().__init__(context, params)
        self.tfidf = params['tfidf']
        self.first_words = params['first_words']

    def fit(self, X, y):
        self.transformer = []
        for col in self.context.text_cols:
            text = [clean_text(s, self.first_words) for s in X[col].values]
            params = self.params
            params.pop('tfidf')
            params.pop('first_words')
            if self.tfidf:
                encoder = TfidfVectorizer(**params)
            else:
                encoder = CountVectorizer(**params)
            encoder.fit(text)
            self.context.feature_names.remove(col)
            self.context.feature_names += [col+'__'+x.replace(' ', '_') for x in encoder.get_feature_names()]
            self.transformer.append((col, encoder))
        # update new list of columns
        self.context.text_cols = []

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            # remove col in X
            text = [clean_text(s, self.first_words) for s in X[col].values]
            T = pd.DataFrame(encoder.transform(text).todense()).reset_index(drop=True)
            T.columns = [col+'__'+x.replace(' ', '_') for x in encoder.get_feature_names()]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            X.drop(col, axis=1, inplace=True)
        return X


class HyperProcessWord2Vec(HyperProcess):
    # class for process word2vec for text

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.transformer = []
        for col in self.context.text_cols:
            encoder = model_word2vec(X[col].values, self.params)
            self.context.feature_names.remove(col)
            self.context.feature_names += [col + '__length'] + [col+'__'+str(i) for i in range(self.params['size'])]
            self.transformer.append((col, encoder))
        # update new list of columns
        self.context.text_cols = []

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_word2vec(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col + '__length'] + [col+'__'+str(i) for i in range(self.params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


class HyperProcessFastText(HyperProcess):
    # class for process fasttext for text

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.transformer = []
        for col in self.context.text_cols:
            encoder = model_fasttext(X[col].values, self.params)
            self.context.feature_names.remove(col)
            self.context.feature_names += [col + '__length'] + [col+'__'+str(i) for i in range(self.params['size'])]
            self.transformer.append((col, encoder))
        # update new list of columns
        self.context.text_cols = []

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_fasttext(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col + '__length'] + [col+'__'+str(i) for i in range(self.params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


class HyperProcessDoc2Vec(HyperProcess):
    # class for process doc2vec for text

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.transformer = []
        for col in self.context.text_cols:
            encoder = model_doc2vec(X[col].values, self.params)
            self.context.feature_names.remove(col)
            self.context.feature_names += [col+'__'+str(i) for i in range(self.params['size'])]
            self.transformer.append((col, encoder))
        # update new list of columns
        self.context.text_cols = []

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_doc2vec(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col+'__'+str(i) for i in range(self.params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


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


class HyperProcessMissingFixed(HyperProcess):
    # class for transformation of missing values with a fixed value

    def __init__(self, context, params):
        super().__init__(context, params)

    def transform(self, X):
        return X.fillna(self.params['fixed'])


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


class HyperProcessNoScaling(HyperProcess):
    # class for no scaling transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.as_matrix()
        else:
            return X


class HyperProcessTruncatedSVD(HyperProcess):
    # class for Truncated SVD feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.params['n_components'] = min(self.params['n_components'], int(np.shape(X)[1]/2))
        self.transformer = TruncatedSVD(**self.params)
        self.transformer.fit(X, y)
        self.context.feature_names = ['SVD_%d' % i for i in range(self.params['n_components'])]


class HyperProcessFastICA(HyperProcess):
    # class for Fast ICA feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.params['n_components'] = min(self.params['n_components'], int(np.shape(X)[1]/2))
        self.transformer = FastICA(**self.params)
        self.transformer.fit(X, y)
        self.context.feature_names = ['ICA_%d' % i for i in range(self.params['n_components'])]


class HyperProcessPCA(HyperProcess):
    # class for PCA feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.params['n_components'] = min(self.params['n_components'], int(np.shape(X)[1]/2))
        self.transformer = PCA(**self.params)
        self.transformer.fit(X, y)
        self.context.feature_names = ['PCA_%d' % i for i in range(self.params['n_components'])]


class HyperProcessPolynomial(HyperProcess):
    # class for Polynomial feature transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def fit(self, X, y):
        self.transformer = PolynomialFeatures(**self.params)
        self.transformer.fit(X)
        self.context.feature_names = self.transformer.get_feature_names(self.context.feature_names)


class HyperProcessSelectFromModel(HyperProcess):
    # class for feature selection
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, context, params):
        super().__init__(context, params)

    @abstractmethod
    def fit(self, X, y):
        self.transformer.fit_transform(X, y)
        support = self.transformer.get_support()
        self.context.feature_names = [f for i, f in enumerate(self.context.feature_names) if support[i]]


class HyperProcessSelectionLinearSVR(HyperProcessSelectFromModel):
    # class for feature selection with SVM model

    def __init__(self, context, params):
        super().__init__(context, params)
        self.transformer = SelectFromModel(LinearSVR(**self.params))


class HyperProcessSelectionRf(HyperProcessSelectFromModel):
    # class for feature selection with Random forest

    def __init__(self, context, params):
        super().__init__(context, params)
        if self.context.problem_type == 'regression':
            self.transformer = SelectFromModel(RandomForestRegressor(**self.params))
        else:
            self.transformer = SelectFromModel(RandomForestClassifier(**self.params))


class HyperProcessPassThrough(HyperProcess):
    # class for No transformation

    def __init__(self, context, params):
        super().__init__(context, params)

    def transform(self, X):
        return X



class NoSampling(object):

    # no re-sampling

    def __init__(self, params):
        self.params = params

    def fit_sample(self, X, y):
        return X, y


