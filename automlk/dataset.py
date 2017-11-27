import pickle
import glob
import pandas as pd
import numpy as np
import datetime
import shutil

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from .config import METRIC_NULL
from .context import get_dataset_folder, get_data_folder, XySet
from .metrics import metric_map
from .graphs import *
from .store import *


def create_dataset(name, domain, description, problem_type, y_col, source, mode, filename_train, metric,
                   other_metrics=[], val_col='index', cv_folds=5, val_col_shuffle=True, sampling=False,
                   holdout_ratio=0.2, filename_test='', filename_cols='', filename_submit='', url='',
                   col_submit=''):
    """
    creates a dataset

    :param name: name of the dataset
    :param domain: domain classification of the dataset (string separated by /)
    :param description: description of the dataset
    :param problem_type: 'regression' or 'classification'
    :param y_col: name of the target column
    :param source: source of the dataset
    :param mode: standard (train set), benchmark (train + test set), competition (train + submit set)
    :param filename_train: file path of the training set
    :param metric: metric to be used to select the best models ('mse', 'rmse', 'log_loss', ...)
    :param other_metrics: secondary metrics, separated by comma (eg: f1, accuracy)
    :param val_col: column name to perform the cross validation (default = 'index')
    :param cv_folds: number of cross validation folds (default = 5)
    :param val_col_shuffle: need to shuffle in cross validation (default = True)
    :param sampling: use re-sampling pre-processing (default = false)
    :param holdout_ratio: holdout ration to split train / eval set
    :param filename_test: name of the test set (benchmark mode)
    :param filename_cols: file to describe columns
    :param filename_submit: name of the submit set (competition mode)
    :param url: url of the dataset
    :param col_submit: index column to be used in submit file (competition mode)
    :return: dataset object
    """

    # create object and control data
    creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    others = [m for m in other_metrics.replace(' ', '').split(',') if m != '']
    dt = DataSet(0, name, domain, description, problem_type, y_col, source, mode, filename_train, metric,
                 other_metrics=others, val_col=val_col, cv_folds=cv_folds,
                 val_col_shuffle=val_col_shuffle, sampling=sampling, holdout_ratio=holdout_ratio,
                 filename_test=filename_test, filename_cols=filename_cols,
                 filename_submit=filename_submit, col_submit=col_submit,
                 url=url, creation_date=creation_date)

    # control data
    df_cols, df_train, df_test, df_submit = dt.initialize_data()

    # update stats
    dt.update_stats(df_train, df_test, df_submit)

    # save and create objects and graphs related to the dataset
    dataset_id = str(incr_key_store('dataset:counter'))
    set_key_store('dataset:%s:status' % dataset_id, 'created')
    set_key_store('dataset:%s:grapher' % dataset_id, False)
    set_key_store('dataset:%s:results' % dataset_id, 0)
    set_key_store('dataset:%s:level' % dataset_id, 1)
    rpush_key_store('dataset:list', dataset_id)

    dt.save(dataset_id)
    dt.create_folders()
    dt.finalize_creation(df_train, df_test, df_submit)

    # create graphs
    __send_grapher(dt.dataset_id)
    return dt


def create_dataset_sets(dt):
    """
    creates the train & test set, and the evaluation set per folds

    :param dt: dataset
    :return:
    """
    # create train & test set
    X, y, X_train, X_test, y_train, y_test, X_submit, id_submit = __create_train_test(dt)

    # prepare y values
    y, y_train, y_test = __prepare_y(dt, y, y_train, y_test)

    # create cv folds
    cv_folds = __create_cv(dt, X_train, y_train)

    # prepare and store eval set
    y_eval_list, y_eval, idx_eval = __store_eval_set(dt, y_train, y_test, cv_folds)

    # then store all these results in a pickle store
    ds = XySet(X, y, X_train, y_train, X_test, y_test, X_submit, id_submit, cv_folds, y_eval_list, y_eval, idx_eval)
    pickle.dump(ds, open(get_dataset_folder(dt.dataset_id) + '/data/eval_set.pkl', 'wb'))


def get_dataset(dataset_id, include_results=False):
    """
    get the descriptive data of a dataset

    :param dataset_id: id of the dataset
    :param include_results: if need to extract results also
    :return: dataset object
    """
    d = get_key_store('dataset:%s' % dataset_id)

    # upward compatibility
    # new fields
    if 'mode' not in d['init_data'].keys():
        if d['init_data']['filename_test'] != '':
            d['init_data']['mode'] = 'benchmark'
        elif d['init_data']['filename_submit'] != '':
            d['init_data']['mode'] = 'competition'
        else:
            d['init_data']['mode'] = 'standard'
    if 'domain' not in d['init_data'].keys():
        d['init_data']['domain'] = ''
    if 'text_cols' not in d['load_data'].keys():
        d['load_data']['text_cols'] = []
    if 'sampling' not in d['init_data'].keys():
        d['init_data']['sampling'] = False
    # deleted fields
    if 'is_public' in d['init_data'].keys():
        d['init_data'].pop('is_public')
    if 'is_uploaded' in d['init_data'].keys():
        d['init_data'].pop('is_uploaded')

    # then load dataset object
    dt = DataSet(**d['init_data'])
    dt.load(d['load_data'], d['features'])

    # add counters and results
    dt.status = get_key_store('dataset:%s:status' % dataset_id)
    dt.grapher = get_key_store('dataset:%s:grapher' % dataset_id)
    dt.round_counter = get_counter_store('dataset:%s:round_counter' % dataset_id)

    if include_results:
        dt.results = get_key_store('dataset:%s:results' % dataset_id)

    return dt


def update_dataset(dataset_id, name, domain, description, is_uploaded, source, url):
    """
    update specific fields of the dataset

    :param dataset_id: id of the dataset
    :param name: new name of the dataset
    :param description: new description of the dataset
    :param is_uploaded: info on uploaded
    :param source: source of the dataset
    :param url: url of the dataset
    :return:
    """
    dt = get_dataset(dataset_id)
    dt.name = name
    dt.domain = domain
    dt.description = description
    dt.is_uploaded = is_uploaded
    dt.source = source
    dt.url = url
    dt.save(dataset_id)


def update_feature_dataset(dataset_id, name, description, to_keep, col_type):
    """
    update specifically some attributes of one column

    :param dataset_id: id of the dataset
    :param name: name of the column
    :param description: new description of the column
    :param to_keep: keep this column
    :param col_type: type of the column
    :return:
    """
    dt = get_dataset(dataset_id)

    # retrieves the feature
    for f in dt.features:
        if f.name == name:
            # update data
            f.description = description
            if to_keep == 'True':
                f.to_keep = True
            else:
                f.to_keep = False
            f.col_type = col_type

    # regenerate list of X columns, categoricals and text columns
    dt.x_cols = [col.name for col in dt.features if col.to_keep and (col.name not in [dt.y_col, dt.val_col])]
    dt.cat_cols = [col.name for col in dt.features if (col.name in dt.x_cols) and (col.col_type == 'categorical')]
    dt.text_cols = [col.name for col in dt.features if (col.name in dt.x_cols) and (col.col_type == 'text')]
    dt.n_cat_cols = len(dt.cat_cols)

    # then save dataset data
    dt.save(dataset_id)


def reset_dataset(dataset_id):
    """
    reset a dataset: erases the search, reset counters and regenerates the graphs

    :param dataset_id: id
    :return:
    """
    root = get_dataset_folder(dataset_id)
    for folder in ['predict', 'submit', 'features', 'models', 'graphs']:
        for f in glob.glob(root + '/' + folder + '/*.*'):
            os.remove(f)

    # reset entries
    set_key_store('dataset:%s:status' % dataset_id, 'created')
    set_key_store('dataset:%s:grapher' % dataset_id, False)
    set_key_store('dataset:%s:results' % dataset_id, 0)
    if exists_key_store('dataset:%s:round_counter' % dataset_id):
        del_key_store('dataset:%s:round_counter' % dataset_id)
    if exists_key_store('dataset:%s:rounds' % dataset_id):
        del_key_store('dataset:%s:rounds' % dataset_id)
    if exists_key_store('dataset:%s:search' % dataset_id):
        del_key_store('dataset:%s:search' % dataset_id)

    # create graphs
    dt = get_dataset(dataset_id)
    df_train = dt.get_data()
    __send_grapher(dt.dataset_id)


def delete_dataset(dataset_id):
    """
    deletes a dataset and the results

    :param dataset_id: id
    :return:
    """
    try:
        shutil.rmtree(get_dataset_folder(dataset_id))
    except:
        pass

    # removes entries
    del_key_store('dataset:%s:status' % dataset_id)
    del_key_store('dataset:%s:results' % dataset_id)
    del_key_store('dataset:%s:grapher' % dataset_id)
    lrem_key_store('dataset:list', dataset_id)
    if exists_key_store('dataset:%s:round_counter' % dataset_id):
        del_key_store('dataset:%s:round_counter' % dataset_id)
    if exists_key_store('dataset:%s:rounds' % dataset_id):
        del_key_store('dataset:%s:rounds' % dataset_id)
    if exists_key_store('dataset:%s:search' % dataset_id):
        del_key_store('dataset:%s:search' % dataset_id)


def __send_grapher(dataset_id):
    """

    :param dataset_id: dataset to request
    :return:
    """
    # send queue the next graph job to do
    msg_search = {'dataset_id': dataset_id}
    print('sending %s' % msg_search)
    lpush_key_store('grapher:queue', msg_search)


def create_graph_data(dataset_id):
    """
    creates the graphs for each column feature of the dataset

    :param dataset_id: dataset id
    :return:
    """
    dataset = get_dataset(dataset_id)
    df = dataset.get_data()

    # create a sample set
    pickle.dump(df.head(20), open(get_dataset_folder(dataset_id) + '/data/sample.pkl', 'wb'))

    # fillna to avoid issues
    for f in dataset.features:
        if f.col_type == 'numerical':
            df[f.name].fillna(0, inplace=True)
        else:
            df[f.name].fillna('', inplace=True)
            df[f.name] = df[f.name].map(str)

    # create graph of target distrubution and correlations
    graph_histogram(dataset_id, dataset.y_col, dataset.is_y_categorical, df[dataset.y_col].values)
    graph_correl_features(dataset, df.copy())

    # create graphs for all features
    for f in dataset.features:
        if f.to_keep and f.name != dataset.y_col:
            print(f.name)
            if f.col_type == 'numerical' and dataset.problem_type == 'regression':
                graph_regression_numerical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'categorical' and dataset.problem_type == 'regression':
                graph_regression_categorical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'numerical' and dataset.problem_type == 'classification':
                graph_classification_numerical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'categorical' and dataset.problem_type == 'classification':
                graph_classification_categorical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'text':
                graph_text(dataset_id, df, f.name)

    # update status grapher
    set_key_store('dataset:%s:grapher' % dataset_id, True)


def get_dataset_sample(dataset_id):
    """
    retrieves a sample of the dataset

    :param dataset_id: dataset id
    :return: list of records as dictionaries
    """
    filename = get_dataset_folder(dataset_id) + '/data/sample.pkl'
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb')).to_dict(orient='records')
    else:
        return [{}]


def get_dataset_list(include_results=False):
    """
    get the list of all datasets

    :param include_status: flag to determine if the status are also retrieved (default = False)
    :return: list of datasets objects or empty list if error (eg. redis or environment not set)
    """
    try:
        return [get_dataset(dataset_id, include_results) for dataset_id in get_dataset_ids()]
    except:
        return []


def get_dataset_ids():
    """
    get the list of ids all datasets

    :return: list of ids
    """
    return list_key_store('dataset:list')


class DataSet(object):
    """
    a dataset is an object managing all the features and data of an experiment
    """

    def __init__(self, dataset_id, name, domain, description, problem_type, y_col, source, mode, filename_train, metric,
                 other_metrics, val_col, cv_folds, val_col_shuffle, sampling,
                 holdout_ratio, filename_test, filename_cols, filename_submit, url, col_submit, creation_date):
        """
        creates a new dataset: it will be automatically stored

        :param name: name of the dataset
        :param domain: domain classification of the dataset (string separated by /)
        :param description: description of the dataset
        :param problem_type: 'regression' or 'classification'
        :param y_col: name of the target column
        :param source: source of the dataset
        :param mode: standard (train set), benchmark (train + test set), competition (train + submit set)
        :param filename_train: file path of the training set
        :param metric: metric to be used to select the best models ('mse', 'rmse', 'log_loss', ...)
        :param other_metrics: list of secondary metrics (as a list)
        :param val_col: column name to perform the cross validation (default = 'index')
        :param cv_folds: number of cross validation folds (default = 5)
        :param val_col_shuffle: need to shuffle in cross validation (default = false)
        :param sampling: use re-sampling pre-processing (default = false)
        :param holdout_ratio: holdout ration to split train / eval set
        :param filename_test: name of the test set
        :param filename_cols: (not implemented)
        :param url: url of the dataset
        """

        # TODO: add feature details from file or dataframe

        # descriptive data:
        self.dataset_id = dataset_id
        self.name = name
        self.domain = domain
        self.description = description

        # problem and optimisation
        if problem_type not in ['regression', 'classification']:
            raise ValueError('problem type must be regression or classification')
        self.problem_type = problem_type

        self.with_test_set = False
        self.url = url
        self.source = source  # url or file id
        self.y_col = y_col
        if filename_train == '':
            raise ValueError('filename train cannot be empty')
        # check // mode
        self.mode = mode
        if mode not in ['standard', 'benchmark', 'competition']:
            raise ValueError('mode must be standard, benchmark or competition')
        if mode == 'standard':
            if filename_test != '':
                raise ValueError('test set should be empty in standard mode')
            if filename_submit != '' or col_submit != '':
                raise ValueError('submit set should be empty in standard mode')
        elif mode == 'benchmark':
            if filename_test == '':
                raise ValueError('test set cannot be empty in benchmark mode')
            if filename_submit != '' or col_submit != '':
                raise ValueError('submit set should be empty in benchmark mode')
        else:
            # competition mode
            if filename_test != '':
                raise ValueError('test set cannot be empty in competition mode')
            if filename_submit == '' or col_submit == '':
                raise ValueError('submit set cannot be empty in competition mode')

        self.filename_train = filename_train
        self.filename_test = filename_test
        self.filename_cols = filename_cols
        self.filename_submit = filename_submit
        self.col_submit = col_submit
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
        self.sampling = sampling
        if (cv_folds < 1) or (cv_folds > 20):
            raise ValueError('cv folds must be in range 1 to 20')
        self.cv_folds = cv_folds
        if (holdout_ratio < 0) or (holdout_ratio > 1):
            raise ValueError('holdout_ratio must be in range 0 to 1')
        self.holdout_ratio = holdout_ratio

    def initialize_data(self):
        # initialize dataframes

        # check column description data
        if self.filename_cols != '':
            self.__check_data(self.filename_cols)
            df_cols = self.__read_data(self.filename_cols)
        else:
            df_cols = pd.DataFrame()

        # check train data
        self.__check_data(self.filename_train)
        df_train = self.__read_data(self.filename_train)
        if self.y_col not in df_train.columns:
            raise ValueError('y_col %s not in the columns of the dataset' % self.y_col)

        self.features, self.is_y_categorical, self.y_n_classes, self.y_class_names = self.__initialize_features(
            df_train, df_cols)

        # check test data
        if self.filename_test != '':
            self.__check_data(self.filename_test)
            df_test = self.__read_data(self.filename_test)
            self.with_test_set = True
            self.holdout_ratio = 0
        else:
            df_test = pd.DataFrame()

        if self.filename_submit != '':
            self.__check_data(self.filename_submit)
            df_submit = self.__read_data(self.filename_submit)
        else:
            df_submit = pd.DataFrame()

        self.x_cols = [col.name for col in self.features if
                       col.to_keep and (col.name not in [self.y_col, self.val_col])]

        self.cat_cols = [col.name for col in self.features if
                         (col.name in self.x_cols) and (col.col_type == 'categorical')]

        self.text_cols = [col.name for col in self.features if
                         (col.name in self.x_cols) and (col.col_type == 'text')]

        self.missing_cols = [col.name for col in self.features if col.n_missing > 0]

        return df_cols, df_train, df_test, df_submit

    def update_stats(self, df_train, df_test, df_submit):
        # create stats on the dataset
        self.size = int(df_train.memory_usage().sum() / 1000000)
        self.n_rows = int(len(df_train) / 1000)
        self.n_cols = len(df_train.columns)
        self.n_cat_cols = len(self.cat_cols)
        self.n_missing = len(self.missing_cols)
        self.test_size = int(df_test.memory_usage().sum() / 1000000)
        self.submit_size = int(df_submit.memory_usage().sum() / 1000000)

    def save(self, dataset_id):
        # saves dataset data in a pickle store
        self.dataset_id = dataset_id

        # save as json
        store = {'init_data': {'dataset_id': self.dataset_id, 'name': self.name, 'domain': self.domain,
                               'description': self.description, 'problem_type': self.problem_type, 'y_col': self.y_col,
                               'source': self.source, 'mode': self.mode,
                               'filename_train': self.filename_train, 'metric': self.metric,
                               'other_metrics': self.other_metrics, 'val_col': self.val_col, 'cv_folds': self.cv_folds,
                               'val_col_shuffle': self.val_col_shuffle, 'sampling': self.sampling,
                               'holdout_ratio': self.holdout_ratio, 'filename_test': self.filename_test,
                               'filename_cols': self.filename_cols, 'url': self.url,
                               'filename_submit': self.filename_submit, 'col_submit': self.col_submit,
                               'creation_date': self.creation_date},
                 'load_data': {'size': self.size, 'n_rows': self.n_rows, 'n_cols': self.n_cols,
                               'n_cat_cols': self.n_cat_cols, 'n_missing': self.n_missing,
                               'with_test_set': self.with_test_set, 'x_cols': self.x_cols, 'text_cols': self.text_cols,
                               'cat_cols': self.cat_cols, 'missing_cols': self.missing_cols,
                               'best_is_min': self.best_is_min, 'is_y_categorical': self.is_y_categorical,
                               'y_n_classes': self.y_n_classes, 'y_class_names': self.y_class_names},
                 'features': [{'name': f.name, 'raw_type': str(f.raw_type), 'n_missing': int(f.n_missing),
                               'n_unique_values': int(f.n_unique_values), 'first_unique_values': f.first_unique_values,
                               'description': f.description, 'to_keep': f.to_keep, 'col_type': f.col_type}
                              for f in self.features]
                 }
        set_key_store('dataset:%s' % self.dataset_id, store)

    def load(self, load_data, features):
        # reload data from json
        for k in load_data.keys():
            setattr(self, k, load_data[k])
        self.features = [Feature(**f) for f in features]

    def finalize_creation(self, df_train, df_test, df_submit):
        # generates objects related to the dataset
        self.__import_data(self.filename_train, 'train')
        if self.filename_test != '':
            self.__import_data(self.filename_test, 'test')
        if self.filename_submit != '':
            self.__import_data(self.filename_submit, 'submit')

    def get_data(self, part='train'):
        """
        returns the imported data of the dataset as a dataframe (when the dataset is created)

        :param part:part of the dataset (train / test)
        :return: data as a dataframe
        """
        return pd.read_pickle(self.__folder() + '/data/%s.pkl' % part)

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
                if metric.binary:
                    y_pred_metric = y_pred[:, 1]
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
                if metric.average and self.y_n_classes > 2:
                    # need average if multi-class
                    return -metric.function(y_act, y_pred_metric, average='micro')
                else:
                    return -metric.function(y_act, y_pred_metric)
        except Exception as e:
            print('error in evaluating metric %s: %s' % (metric_name, e))
            return METRIC_NULL

    def create_folders(self):
        # create folders
        root = get_data_folder() + '/%s' % self.dataset_id
        os.makedirs(root)
        os.makedirs(root + '/data')
        os.makedirs(root + '/predict')
        os.makedirs(root + '/submit')
        os.makedirs(root + '/features')
        os.makedirs(root + '/models')
        os.makedirs(root + '/graphs')
        os.makedirs(root + '/graphs_dark')

    def __folder(self):
        # storage folder of the dataset
        return get_dataset_folder(self.dataset_id)

    def __check_data(self, filename, part='train'):
        # check data file in the dataset

        ext = filename.split('.')[-1].lower()
        if ext not in ['csv', 'tsv', 'xls', 'xlsx']:
            raise TypeError('unknown dataset format: use csv, xls or xlsx')

        if not os.path.exists(filename):
            raise ValueError('file %s not found' % filename)

    def __read_data(self, filename):
        # read the dataset (without import)

        ext = filename.split('.')[-1]
        if ext == 'csv':
            df = pd.read_csv(filename)
        elif ext == 'tsv':
            df = pd.read_csv(filename, sep='\t', header=0)
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(filename)
        return df

    def __initialize_features(self, df, df_cols):
        # creates the columns for a dataset from the data as a dataframe

        cols = {x['name']: x for x in df_cols.fillna('').to_dict(orient='records')}

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
            n_unique_ratio = n_unique / len(df)
            raw_type = str(df[col].dtype)
            first_unique_values = ', '.join([str(x) for x in uniques[:5]])
            to_keep = True
            description = ''
            col_type = ''
            if col in cols:
                k = cols[col]
                description = k['description']
                if k['to_keep'] != '':
                    to_keep = k['to_keep']
                if k['col_type'] != '':
                    col_type = k['col_type']

            is_y = (col == self.y_col)
            feature = Feature(col, raw_type, n_missing, n_unique, first_unique_values, description, to_keep, col_type,
                              n_unique_ratio, is_y)
            features.append(feature)

            if col == self.y_col:
                if (self.problem_type == 'regression') and (feature.col_type == 'categorical'):
                    raise ValueError('target column %s must be numerical in regression' % col)

                is_y_categorical = (feature.col_type == 'categorical')
                y_n_classes = int(feature.n_unique_values)
                y_class_names = [str(x) for x in np.sort(uniques)]

        return features, is_y_categorical, y_n_classes, y_class_names

    def __import_data(self, filename, part):
        # copy file in the dataset
        df = self.__read_data(filename)
        # save as pickle
        df.to_pickle(self.__folder() + '/data/%s.pkl' % part)


class Feature(object):
    def __init__(self, name, raw_type, n_missing, n_unique_values, first_unique_values, description, to_keep,
                 col_type, n_unique_ratio=0, is_y=False):
        # descriptive data
        self.name = name
        self.description = description
        self.to_keep = to_keep
        self.raw_type = raw_type

        self.n_missing = n_missing
        self.n_unique_values = n_unique_values
        self.n_unique_ratio = n_unique_ratio
        self.first_unique_values = first_unique_values
        self.print_values = ', '.join([str(x) for x in self.first_unique_values])

        # initialize type
        if col_type != '':
            if col_type not in ['numerical', 'categorical', 'text']:
                raise ValueError('feature %s col type: %s should be numerical, categorical or text' % (name, col_type))
            self.col_type = col_type
        else:
            if raw_type.startswith('float'):
                self.col_type = 'numerical'
            elif raw_type.startswith('int'):
                self.col_type = 'numerical'
            else:
                # raw_type in ['str', 'object']:
                self.col_type = 'categorical'

            # additional rules
            if not is_y:
                if self.col_type == 'numerical' and self.n_unique_values < 10:
                    self.col_type = 'categorical'
                if self.col_type == 'categorical' and self.n_unique_ratio > 0.5:
                    self.col_type = 'text'

            # TODO : manage dates


def __create_train_test(dataset):
    dataset = get_dataset(dataset.dataset_id)
    data_train = dataset.get_data()
    # TODO: split according to val_col
    # split into train & test set
    if dataset.with_test_set:
        data_test = dataset.get_data('test')

        X_train = data_train[dataset.x_cols]
        y_train = data_train[dataset.y_col].values

        X_test = data_test[dataset.x_cols]
        y_test = data_test[dataset.y_col].values

        X = X_train.copy()
        y = y_train.copy()
    else:
        X = data_train[dataset.x_cols]
        y = data_train[dataset.y_col].values

        # test set generated by split with holdout ratio
        if dataset.problem_type == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.holdout_ratio,
                                                                shuffle=dataset.val_col_shuffle, random_state=0)
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.holdout_ratio,
                                                                    shuffle=dataset.val_col_shuffle,
                                                                    stratify=y,
                                                                    random_state=0)
            except:
                # may fail if two few classes -> split without stratify
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dataset.holdout_ratio,
                                                                    shuffle=dataset.val_col_shuffle,
                                                                    random_state=0)

    # submit data
    if dataset.filename_submit != '':
        df = dataset.get_data('submit')
        X_submit = df[dataset.x_cols]
        id_submit = df[dataset.col_submit].values
    else:
        X_submit = []
        id_submit = []

    return X, y, X_train, X_test, y_train, y_test, X_submit, id_submit


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


def __prepare_y(dataset, y, y_train, y_test):
    # pre-processing of y: categorical
    if dataset.problem_type == 'classification':
        # encode class values as integers
        encoder = LabelEncoder()
        #encoder.fit([str(x) for x in np.concatenate((y_train, y_test), axis=0)])
        y = encoder.fit_transform(y)
        y_train = encoder.transform([str(x) for x in y_train])
        y_test = encoder.transform([str(x) for x in y_test])
    return y, y_train, y_test


def get_y_eval(dataset_id):
    """
    retrieves the y value of the eval set

    :param dataset_id: id of the dataset
    :return: y values
    """
    #
    return pickle.load(open(get_dataset_folder(dataset_id) + '/y_eval.pkl', 'rb'))


