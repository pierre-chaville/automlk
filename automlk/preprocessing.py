from abc import ABCMeta, abstractmethod
import numpy as np
import pickle
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Imputer, PolynomialFeatures, \
    LabelEncoder
from sklearn.decomposition import TruncatedSVD, FastICA, PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .spaces.process import *
from .utils.text_encoders import *
from .context import text_model_filename

try:
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument

    import_gensim = True
except:
    import_gensim = False


class Transformer(object):
    __metaclass__ = ABCMeta

    # abstract class for model preprocessing in hyper optimisation

    @abstractmethod
    def __init__(self, **params):
        self.set_params(**params)
        self.transformer = None
        self.feature_names = []
        self.info = ''
        self.url = ''

    @abstractmethod
    def set_params(self, **params):
        self.params = params
        self.context = params['context']
        self.transformer_params = {key: params[key] for key in params.keys() if key not in ['context']}
        self.details = []

    @abstractmethod
    def get_params(self, deep=True):
        return self.params

    @abstractmethod
    def get_feature_names(self):
        return self.feature_names

    @abstractmethod
    def fit(self, X, y):
        # fit the transformer with the data
        self.feature_names = list(X.columns)

    @abstractmethod
    def transform(self, X):
        # transform
        Xt = self.transformer.transform(X)
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
            Xt.columns = self.feature_names
        return Xt

    @abstractmethod
    def fit_transform(self, X, y):
        # fit and transform
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def get_details(self):
        # additional details on the transformation process
        return self.details


class TransformerMissingFixed(Transformer):
    # class for transformation of missing values with a fixed value

    def __init__(self, **params):
        super().__init__(**params)
        self.missing = [f['name'] for f in self.context if f['n_missing'] > 0]
        self.details = self.missing

    def transform(self, X):
        for col in self.missing:
            # add new column indicator col__isnulll
            X[col + '__isnull'] = X[col].isnull()
            # fill missing columns only with fixed value
            X[col] = X[col].fillna(self.transformer_params['fixed'])
        # fill potential residual NaN (eg in new predictions)
        return X.fillna(0)


class TransformerMissingFrequency(Transformer):
    # class for transformation of missing values depending on the missing frequency ratio

    def __init__(self, **params):
        super().__init__(**params)
        self.missing = [f for f in self.context if f['n_missing'] > 0]

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.transformer = []
        for f in self.missing:
            col = f['name']
            missing_ratio = X[col].isnull().sum() / len(X)
            if missing_ratio < self.transformer_params['frequency']:
                df = X[[col]].dropna()
                # replace by most frequent
                val = df[col].value_counts().idxmax()
                self.details.append(('frequent', f['col_type'], col, val))
            else:
                if f['col_type'] == 'categorical':
                    val = 'NAN'
                elif f['col_type'] == 'text':
                    val = ''
                elif f['col_type'] == 'numerical':
                    values = X[col].dropna().values
                    val_min = min(values)
                    if val_min > 0:
                        val = 0
                    elif val_min > -1:
                        val = -1
                    else:
                        val = -1000
                else:
                    val = -1
                self.details.append(('inf', f['col_type'], col, val))
            self.transformer.append((col, val))

    def transform(self, X):
        for col, val in self.transformer:
            # add new column indicator col__isnulll
            X[col + '__isnull'] = X[col].isnull()
            # fill missing columns only with specific value for missing column
            X[col] = X[col].fillna(val)
        # fill potential residual NaN (eg in new predictions)
        return X.fillna(0)


class TransformerCategorical(Transformer):
    # class for process categorical encoding

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        super().__init__(**params)
        self.cat_cols = [f['name'] for f in self.context if f['col_type'] == 'categorical']
        self.details = self.cat_cols

    @abstractmethod
    def fit(self, X, y):
        self.transformer.fit(X, y)
        # update new list of columns
        Xt = self.transformer.transform(X.copy())
        self.feature_names = list(Xt.columns)


class TransformerLabel(TransformerCategorical):
    # class for process categorical encoding - label encoder

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = []

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.transformer = []
        for col in self.cat_cols:
            encoder = {x: i for i, x in enumerate(X[col].unique())}
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            X[col] = X[col].map(lambda x: encoder[x] if x in encoder else -1)
        return X


class TransformerOneHot(TransformerCategorical):
    # class for process categorical encoding - one hot

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = ce.OneHotEncoder(**self.transformer_params)


class TransformerBaseN(TransformerCategorical):
    # class for process categorical encoding - base N

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = ce.BaseNEncoder(**self.transformer_params)


class TransformerHashing(TransformerCategorical):
    # class for process categorical encoding - hashing

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = ce.HashingEncoder(**self.transformer_params)


class TransformerBOW(Transformer):
    # class for process bag of words for text

    def __init__(self, **params):
        super().__init__(**params)
        self.text_cols = [f['name'] for f in self.context if f['col_type'] == 'text']

    def fit(self, X, y):
        self.transformer = []
        self.feature_names = list(X.columns)
        for col in self.text_cols:
            encoder = get_text_encoder(self.context, col, 'bow', self.transformer_params)
            if encoder is None:
                encoder = model_bow(X[col].values, self.transformer_params)
            self.feature_names.remove(col)
            self.feature_names += [col + '__' + x.replace(' ', '_') for x in encoder.get_feature_names()]
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            # remove col in X
            text = [clean_text(s) for s in X[col].values]
            T = pd.DataFrame(encoder.transform(text).todense()).reset_index(drop=True)
            T.columns = [col + '__' + x.replace(' ', '_') for x in encoder.get_feature_names()]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            X.drop(col, axis=1, inplace=True)
        return X


class TransformerWord2Vec(Transformer):
    # class for process word2vec for text

    def __init__(self, **params):
        super().__init__(**params)
        self.text_cols = [f['name'] for f in self.context if f['col_type'] == 'text']
        self.details = self.text_cols

    def fit(self, X, y):
        self.transformer = []
        self.feature_names = list(X.columns)
        for col in self.text_cols:
            encoder = get_text_encoder(self.context, col, 'w2v', self.transformer_params)
            if encoder is None:
                encoder = model_word2vec(X[col].values, self.transformer_params)
            self.feature_names.remove(col)
            self.feature_names += [col + '__length'] + [col + '__' + str(i) for i in
                                                        range(self.transformer_params['size'])]
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_word2vec(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col + '__length'] + [col + '__' + str(i) for i in range(self.transformer_params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


class TransformerFastText(Transformer):
    # class for process fasttext for text

    def __init__(self, **params):
        super().__init__(**params)
        self.text_cols = [f['name'] for f in self.context if f['col_type'] == 'text']
        self.details = self.text_cols

    def fit(self, X, y):
        self.transformer = []
        self.feature_names = list(X.columns)
        for col in self.text_cols:
            encoder = model_fasttext(X[col].values, self.transformer_params)
            self.feature_names.remove(col)
            self.feature_names += [col + '__length'] + [col + '__' + str(i) for i in
                                                        range(self.transformer_params['size'])]
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_fasttext(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col + '__length'] + [col + '__' + str(i) for i in range(self.transformer_params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


class TransformerDoc2Vec(Transformer):
    # class for process doc2vec for text

    def __init__(self, **params):
        super().__init__(**params)
        self.text_cols = [f['name'] for f in self.context if f['col_type'] == 'text']
        self.details = self.text_cols

    def fit(self, X, y):
        self.transformer = []
        self.feature_names = list(X.columns)
        for col in self.text_cols:
            encoder = get_text_encoder(self.context, col, 'd2v', self.transformer_params)
            if encoder is None:
                encoder = model_doc2vec(X[col].values, self.transformer_params)
            self.feature_names.remove(col)
            self.feature_names += [col + '__' + str(i) for i in range(self.transformer_params['size'])]
            self.transformer.append((col, encoder))

    def transform(self, X):
        # transform X
        for col, encoder in self.transformer:
            T = pd.DataFrame(vector_doc2vec(encoder, X[col].values, self.params)).reset_index(drop=True)
            T.columns = [col + '__' + str(i) for i in range(self.transformer_params['size'])]
            X = pd.concat([X.reset_index(drop=True), T], axis=1)
            # remove col in X
            X.drop(col, axis=1, inplace=True)
        return X


def get_text_encoder(features, col, model_type, params):
    """
    get the encoder from a text column (col), with params

    :param features: list of features of the dataset
    :param col: column name
    :param model_type: model type (bow, w2v, d2v)
    :param params: params of the encoder (size, ...)
    :return: encoder or None
    """
    for f in features:
        if f['name'] == col:
            ref = f['text_ref']
            if ref != '':
                filename = text_model_filename(ref, model_type, params)
                return pickle.load(open(filename, 'rb'))
            else:
                return None
    return None


class TransformerScaling(Transformer):
    # abstract class for scaling transformation

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.transformer.fit(X, y)


class TransformerScalingStandard(TransformerScaling):
    # class for scaling

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = StandardScaler(**self.transformer_params)
        self.info = "Standardize features by removing the mean and scaling to unit variance. " \
                    "This scaler can also be applied to sparse CSR or CSC matrices by passing with_mean=False " \
                    "to avoid breaking the sparsity structure of the data."
        self.url = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler"


class TransformerScalingMinMax(TransformerScaling):
    # class for scaling

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = MinMaxScaler(**self.transformer_params)
        self.info = "This estimator scales and translates each feature individually " \
                    "such that it is in the given range on the training set, i.e. between zero and one."
        self.url = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler"


class TransformerScalingMaxAbs(TransformerScaling):
    # class for scaling

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = MaxAbsScaler(**self.transformer_params)
        self.info = "This estimator scales and translates each feature individually " \
                    "such that the maximal absolute value of each feature in the training set will be 1.0. " \
                    "It does not shift/center the data, and thus does not destroy any sparsity." \
                    "This scaler can also be applied to sparse CSR or CSC matrices."
        self.url = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler"


class TransformerScalingRobust(TransformerScaling):
    # class for scaling

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = RobustScaler(**self.transformer_params)
        self.info = "This Scaler removes the median and scales the data according to the quantile range " \
                    "(defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile " \
                    "(25th quantile) and the 3rd quartile (75th quantile)."
        self.url = "http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler"


class TransformerTruncatedSVD(Transformer):
    # class for Truncated SVD feature transformation

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y):
        self.transformer_params['n_components'] = max(2, min(self.transformer_params['n_components'],
                                                             int(len(X.columns) / 2)))
        self.transformer = TruncatedSVD(**self.transformer_params)
        self.transformer.fit(X, y)
        self.feature_names = ['SVD_%d' % i for i in range(self.transformer_params['n_components'])]


class TransformerFastICA(Transformer):
    # class for Fast ICA feature transformation

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y):
        self.transformer_params['n_components'] = max(2, min(self.transformer_params['n_components'],
                                                             int(len(X.columns) / 2)))
        self.transformer = FastICA(**self.transformer_params)
        self.transformer.fit(X, y)
        self.feature_names = ['ICA_%d' % i for i in range(self.transformer_params['n_components'])]


class TransformerPCA(Transformer):
    # class for PCA feature transformation

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y):
        self.transformer_params['n_components'] = max(2, min(self.transformer_params['n_components'],
                                                             int(len(X.columns) / 2)))
        self.transformer = PCA(**self.transformer_params)
        self.transformer.fit(X, y)
        self.feature_names = ['PCA_%d' % i for i in range(self.transformer_params['n_components'])]


class TransformerSelectFromModel(Transformer):
    # class for feature selection
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, **params):
        super().__init__(**params)

    @abstractmethod
    def fit(self, X, y):
        self.transformer.fit_transform(X, y)
        support = self.transformer.get_support()
        self.feature_names = [f for i, f in enumerate(X.columns) if support[i]]


class TransformerSelectionLinearSVR(TransformerSelectFromModel):
    # class for feature selection with SVM model

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = SelectFromModel(LinearSVR(**self.transformer_params))


class TransformerSelectionRfR(TransformerSelectFromModel):
    # class for feature selection with Random forest

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = SelectFromModel(RandomForestRegressor(**self.transformer_params))


class TransformerSelectionRfC(TransformerSelectFromModel):
    # class for feature selection with Random forest

    def __init__(self, **params):
        super().__init__(**params)
        self.transformer = SelectFromModel(RandomForestClassifier(**self.transformer_params))


class TransformerPassThrough(Transformer):
    # class for No transformation

    def __init__(self, **params):
        super().__init__(**params)

    def fit(self, X, y):
        self.feature_names = list(X.columns)

    def transform(self, X):
        return X


class NoSampling(object):

    # no re-sampling

    def __init__(self, **params):
        self.params = params

    def fit_sample(self, X, y):
        return X, y
