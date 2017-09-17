import pickle
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .metrics import metric_map
from .context import METRIC_NULL, get_dataset_folder, get_data_folder
from .graphs import graph_correl_features, graph_histogram
from .store import *


def create_dataset(name, description, problem_type, y_col, is_uploaded, source, filename_train, metric,
                   other_metrics=[], val_col='index', cv_folds=5, val_col_shuffle=True,
                   holdout_ratio=0.2, filename_test='', filename_cols='', is_public=False, url=''):
    # create object and control data
    creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dt = DataSet(0, name, description, problem_type, y_col, is_uploaded, source, filename_train, metric,
                 other_metrics=other_metrics, val_col=val_col, cv_folds=cv_folds, val_col_shuffle=val_col_shuffle,
                 holdout_ratio=holdout_ratio, filename_test=filename_test, filename_cols=filename_cols,
                 is_public=is_public, url=url, creation_date=creation_date)

    # control data
    df_train, df_test = dt.initialize_data()

    # update stats
    dt.update_stats(df_train, df_test)

    # save and create objects and graphs related to the dataset
    dataset_id = str(incr_key_store('dataset:counter'))
    rpush_key_store('dataset:list', dataset_id)

    dt.save(dataset_id)
    dt.finalize_creation(df_train, df_test)

    # create train & test set
    X_train, X_test, y_train, y_test = __create_train_test(dt)

    # prepare y values
    y_train, y_test = __prepare_y(dt, y_train, y_test)

    # create cv folds
    cv_folds = __create_cv(dt, X_train, y_train)

    # prepare and store eval set
    y_eval_list, y_eval, idx_eval = __store_eval_set(dt, y_train, y_test, cv_folds)

    # then store all these results in a pickle store
    pickle.dump([X_train, X_test, y_train, y_test, cv_folds, y_eval_list, y_eval, idx_eval],
                open(get_dataset_folder(dt.dataset_id) + '/data/eval_set.pkl', 'wb'))

    return dt


class DataSet(object):
    """
    a dataset is an object managing all the features and data of an experiment
    """

    def __init__(self, dataset_id, name, description, problem_type, y_col, is_uploaded, source, filename_train, metric,
                 other_metrics, val_col, cv_folds, val_col_shuffle,
                 holdout_ratio, filename_test, filename_cols, is_public, url, creation_date):
        """
        creates a new dataset: it will be automatically stored

        :param name: name of the dataset
        :param description: description of the dataset
        :param problem_type: 'regression' or 'classification'
        :param y_col: name of the target column
        :param is_uploaded: not used
        :param source: source of the dataset
        :param filename_train: file path of the training set
        :param metric: metric to be used to select the best models ('mse', 'rmse', 'log_loss', ...)
        :param other_metrics: list of secondary metrics
        :param val_col: column name to perform the cross validation (default = 'index')
        :param cv_folds: number of cross validation folds (default = 5)
        :param val_col_shuffle: need to shuffle in cross validation (default = false)
        :param holdout_ratio: holdout ration to split train / eval set
        :param filename_test: name of the test set
        :param filename_cols: (not implemented)
        :param is_public: is the dataset in the public domain (true / false)
        :param url: url of the dataset
        """

        # TODO: add feature details from file or dataframe

        # descriptive data:
        self.dataset_id = dataset_id
        self.name = name
        self.description = description

        # problem and optimisation
        if problem_type not in ['regression', 'classification']:
            raise ValueError('problem type must be regression or classification')
        self.problem_type = problem_type

        self.is_public = is_public
        self.with_test_set = False
        self.url = url
        self.is_uploaded = is_uploaded
        self.source = source  # url or file id
        self.y_col = y_col
        self.filename_train = filename_train
        self.filename_test = filename_test
        self.filename_cols = filename_cols
        self.creation_date = creation_date

        if metric not in metric_map.keys():
            raise ValueError('metric %s is not known' % metric)
        if metric_map[metric].problem_type != self.problem_type:
            raise ValueError('metric %s is not compatible with %s' % (metric, problem_type))
        self.metric = metric

        self.best_is_min = metric_map[metric].best_is_min

        for m in other_metrics:
            if m not in metric_map.keys():
                raise ValueError('other metric %s is not known' % m)
            if metric_map[m].problem_type != self.problem_type:
                raise ValueError('metric %s is not compatible with %s' % (m, problem_type))
        self.other_metrics = other_metrics

        # validation & cross validation
        self.val_col = val_col
        self.val_col_shuffle = val_col_shuffle
        if (cv_folds < 1) or (cv_folds > 20):
            raise ValueError('cv folds must be in range 1 to 20')
        self.cv_folds = cv_folds
        if (holdout_ratio < 0) or (holdout_ratio > 1):
            raise ValueError('holdout_ratio must be in range 0 to 1')
        self.holdout_ratio = holdout_ratio

    def initialize_data(self):
        # check train data
        self.__check_data(self.filename_train)
        df_train = self.__read_data(self.filename_train)
        if self.y_col not in df_train.columns:
            raise ValueError('y_col %s not in the columns of the dataset' % self.y_col)

        self.features, self.is_y_categorical, self.y_n_classes, self.y_class_names = self.__initialize_features(
            df_train)

        # check test data
        if self.filename_test != '':
            self.__check_data(self.filename_test)
            df_test = self.__read_data(self.filename_test)
            self.with_test_set = True
            self.holdout_ratio = 0
        else:
            df_test = pd.DataFrame()

        self.x_cols = [col.name for col in self.features if
                       col.to_keep and (col.name not in [self.y_col, self.val_col])]

        self.cat_cols = [col.name for col in self.features if
                         (col.name in self.x_cols) and (col.col_type == 'categorical')]

        self.missing_cols = [col.name for col in self.features if col.n_missing > 0]

        return df_train, df_test

    def update_stats(self, df_train, df_test):
        # create stats on the dataset
        self.size = int(df_train.memory_usage().sum() / 1000000)
        self.n_rows = int(len(df_train) / 1000)
        self.n_cols = len(df_train.columns)
        self.n_cat_cols = len(self.cat_cols)
        self.n_missing = len(self.missing_cols)

    def save(self, dataset_id):
        # saves dataset data in a pickle store
        self.dataset_id = dataset_id
        self.__create_folders()
        pickle.dump(self, open(self.__folder() + '/dataset.pkl', 'wb'))
        # save as json
        store = {'init_data': {'dataset_id': self.dataset_id, 'name': self.name, 'description': self.description,
                               'problem_type': self.problem_type, 'y_col': self.y_col,
                               'is_uploaded': self.is_uploaded, 'source': self.source,
                               'filename_train': self.filename_train, 'metric': self.metric,
                               'other_metrics': self.other_metrics, 'val_col': self.val_col, 'cv_folds': self.cv_folds,
                               'val_col_shuffle': self.val_col_shuffle,
                               'holdout_ratio': self.holdout_ratio, 'filename_test': self.filename_test,
                               'filename_cols': self.filename_cols, 'is_public': self.is_public, 'url': self.url,
                               'creation_date': self.creation_date},
                 'load_data': {'size': self.size, 'n_rows': self.n_rows, 'n_cols': self.n_cols,
                               'n_cat_cols': self.n_cat_cols, 'n_missing': self.n_missing,
                               'with_test_set': self.with_test_set, 'x_cols': self.x_cols,
                               'cat_cols': self.cat_cols, 'missing_cols': self.missing_cols,
                               'self.best_is_min': self.best_is_min, 'is_y_categorical': self.is_y_categorical,
                               'y_n_classes': self.y_n_classes, 'y_class_names': self.y_class_names},
                 'features': [{'name': f.name, 'raw_type': str(f.raw_type), 'n_missing': int(f.n_missing),
                               'n_unique_values': int(f.n_unique_values), 'first_unique_values': f.first_unique_values}
                              for f in self.features]
                 }
        set_key_store('dataset:%s' % self.dataset_id, store)

    def load(self, load_data, features):
        # reload data from json
        for k in load_data.keys():
            setattr(self, k, load_data[k])
        self.features = [Feature(**f) for f in features]

    def finalize_creation(self, df_train, df_test):
        # generates objects related to the dataset
        self.__import_data(self.filename_train, 'train')
        if self.filename_test != '':
            self.__import_data(self.filename_test, 'test')

        self.__create_graphs(df_train, 'train')
        # TODO: create train & test set, cv folds & eval set
        if self.filename_test != '':
            self.__create_graphs(df_test, 'test')

    def get_data(self, part='train'):
        """
        returns the imported data of the dataset as a dataframe (when the dataset is created)

        :param part:part of the dataset (train / test)
        :return: data as a dataframe
        """
        return pd.read_pickle(self.__folder() + '/data/%s.pkl' % part)

    def __folder(self):
        # storage folder of the dataset
        return get_dataset_folder(self.dataset_id)

    def __check_data(self, filename, part='train'):
        # check data file in the dataset

        ext = filename.split('.')[-1].lower()
        if ext not in ['csv', 'xls', 'xlsx']:
            raise TypeError('unknown dataset format: use csv, xls or xlsx')

        if not os.path.exists(filename):
            raise ValueError('file %s not found' % filename)

    def __read_data(self, filename):
        # read the dataset (without import)

        ext = filename.split('.')[-1]
        if ext == 'csv':
            df = pd.read_csv(filename)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(filename)
        return df

    def __initialize_features(self, df):
        # creates the columns for a dataset from the data as a dataframe

        # reset columns
        features = []
        is_y_categorical = False
        y_n_classes = 0
        y_class_names = []

        # retrieve column info from dataframe data
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            uniques = df[col].unique()
            n_unique = len(uniques)
            raw_type = str(df[col].dtype)
            first_unique_values = ', '.join([str(x) for x in uniques[:5]])
            feature = Feature(col, raw_type, n_missing, n_unique, first_unique_values)
            features.append(feature)

            # y_col : categorical if classification, numerical if regression
            if col == self.y_col:
                if (self.problem_type == 'regression') and (feature.col_type == 'categorical'):
                    raise ValueError('target column %s must be numerical in regression' % col.name)

                is_y_categorical = (feature.col_type == 'categorical')
                y_n_classes = int(feature.n_unique_values)
                y_class_names = [str(x) for x in np.sort(uniques)]

        return features, is_y_categorical, y_n_classes, y_class_names

    def evaluate_metric(self, y_act, y_pred, metric_name=None):
        # evaluate score with the metric of the dataset
        if not metric_name:
            metric_name = self.metric
        else:
            if metric_name not in self.other_metrics:
                raise ValueError('evaluation metric not listed in other metrics')
        metric = metric_map[metric_name]
        try:
            if metric.need_class:
                # convert proba to classes
                y_pred_metric = np.argmax(y_pred, axis=1)
            else:
                y_pred_metric = y_pred

            # use sign before metrics to always compare best is min in comparisons
            # but requires to display abs value for display
            if metric.best_is_min:
                if metric.name == 'log_loss':
                    return metric.function(y_act, y_pred_metric, labels=list(range(self.y_n_classes)))
                else:
                    return metric.function(y_act, y_pred_metric)
            else:
                return -metric.function(y_act, y_pred_metric)
        except Exception as e:
            print('error in evaluating metric %s: %s' % (metric_name, e))
            return METRIC_NULL

    def __create_folders(self):
        # create folders
        root = get_data_folder() + '/%s' % self.dataset_id
        os.makedirs(root)
        os.makedirs(root + '/data')
        os.makedirs(root + '/predict')
        os.makedirs(root + '/features')
        os.makedirs(root + '/models')
        os.makedirs(root + '/graphs')
        # initialize search file
        with open(root + '/search.txt', 'w') as f:
            f.write('')

    def __import_data(self, filename, part):
        # copy file in the dataset
        df = self.__read_data(filename)
        # save as pickle
        df.to_pickle(self.__folder() + '/data/%s.pkl' % part)

    def __create_graphs(self, df, part):
        # creates the various graphs of the dataset
        graph_histogram(self.dataset_id, self.y_col, self.is_y_categorical, df[self.y_col].values, part)

        if part == 'train':
            # histogram of the target columns
            graph_correl_features(self, df)

            # histograpm for each feature

    # TODO: implement update dataset
    # TODO: implement delete dataset


class Feature(object):
    def __init__(self, name, raw_type, n_missing, n_unique_values, first_unique_values):
        # descriptive data
        self.name = name
        self.description = ''
        self.to_keep = True
        self.raw_type = raw_type

        # initialize type
        if raw_type.startswith('float'):
            self.col_type = 'numerical'
        elif raw_type.startswith('int'):
            self.col_type = 'numerical'
        elif raw_type in ['str', 'object']:
            self.col_type = 'categorical'
        # TODO : manage dates and text

        self.n_missing = n_missing
        self.n_unique_values = n_unique_values
        self.first_unique_values = first_unique_values
        self.print_values = ', '.join([str(x) for x in self.first_unique_values])


def __create_train_test(dataset):
    print('loading train set')
    dataset = get_dataset(dataset.dataset_id)
    data_train = dataset.get_data()
    # TODO: split according to val_col
    # split into train & test set
    if dataset.with_test_set:
        print('loading test set')
        data_test = dataset.get_data('test')

        X_train = data_train[dataset.x_cols]
        y_train = data_train[dataset.y_col].values

        X_test = data_test[dataset.x_cols]
        y_test = data_test[dataset.y_col].values
    else:
        X = data_train[dataset.x_cols]
        y = data_train[dataset.y_col].values

        # test set generated by split with holdout ratio
        if dataset.problem_type == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.holdout_ratio,
                                                                shuffle=dataset.val_col_shuffle, random_state=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.holdout_ratio,
                                                                shuffle=dataset.val_col_shuffle,
                                                                stratify=y,
                                                                random_state=0)
    return X_train, X_test, y_train, y_test


def __create_cv(dataset, X_train, y_train):
    # generate cv folds
    # TODO: split according to val_col
    if dataset.problem_type == 'classification':
        skf = StratifiedKFold(n_splits=dataset.cv_folds, shuffle=dataset.val_col_shuffle, random_state=0)
        cv_folds = [(train_index, eval_index) for train_index, eval_index in skf.split(X_train, y_train)]
    else:
        skf = KFold(n_splits=dataset.cv_folds, shuffle=dataset.val_col_shuffle, random_state=0)
        cv_folds = [(train_index, eval_index) for train_index, eval_index in skf.split(X_train)]
    return cv_folds


def __store_eval_set(dataset, y_train, y_test, cv_folds):
    y_eval_list = []
    i_eval_list = []
    # stores eval set
    for i, (train_index, eval_index) in enumerate(cv_folds):
        pickle.dump(y_train[train_index], open(get_dataset_folder(dataset.dataset_id) + '/y_train_%d.pkl' % i, 'wb'))
        pickle.dump(y_train[eval_index], open(get_dataset_folder(dataset.dataset_id) + '/y_eval_%d.pkl' % i, 'wb'))
        y_eval_list.append(y_train[eval_index])
        i_eval_list.append(eval_index)

    # stores test set
    pickle.dump(y_test, open(get_dataset_folder(dataset.dataset_id) + '/y_test.pkl', 'wb'))

    # generate y_eval
    y_eval = np.concatenate(y_eval_list, axis=0)

    # store y_eval
    pickle.dump(y_train, open(get_dataset_folder(dataset.dataset_id) + '/y_eval.pkl', 'wb'))

    return y_eval_list, y_eval, np.concatenate(i_eval_list, axis=0)


def __prepare_y(dataset, y_train, y_test):
    # pre-processing of y: categorical
    if dataset.problem_type == 'classification':
        print('y label encoding')
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit([str(x) for x in np.concatenate((y_train, y_test), axis=0)])
        y_train = encoder.transform([str(x) for x in y_train])
        y_test = encoder.transform([str(x) for x in y_test])
    return y_train, y_test


def get_y_eval(dataset_id):
    """
    retrieves the y value of the eval set

    :param dataset_id: id of the dataset
    :return: y values
    """
    #
    return pickle.load(open(get_dataset_folder(dataset_id) + '/y_eval.pkl', 'rb'))


def get_dataset_list():
    """
    get the list of all datasets

    :return: list of datasets objects
    """
    return [get_dataset(dataset_id) for dataset_id in get_dataset_ids()]


def get_dataset_ids():
    """
    get the list of ids all datasets

    :return: list of ids
    """
    return list_key_store('dataset:list')


def get_dataset(dataset_id):
    """
    get the descriptive data of a dataset

    :param dataset_id: id of the dataset
    :return: dataset object
    """
    d = get_key_store('dataset:%s' % dataset_id)
    dt = DataSet(**d['init_data'])
    dt.load(d['load_data'], d['features'])
    return dt
