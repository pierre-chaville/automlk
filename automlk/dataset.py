import logging
import pickle
import glob
import datetime
import shutil
import pandas as pd
import numpy as np
from .metrics import metric_map
from .store import *
from .textset import get_textset_list
from .context import get_dataset_folder


log = logging.getLogger(__name__)


def get_dataset_ids():
    """
    get the list of ids all datasets

    :return: list of ids
    """
    return list_key_store('dataset:list')


def get_dataset_list(include_results=False):
    """
    get the list of all datasets

    :param include_results: flag to determine if the status are also retrieved (default = False)
    :return: list of datasets objects or empty list if error (eg. redis or environment not set)
    """
    # try:
    return [get_dataset(dataset_id, include_results) for dataset_id in get_dataset_ids()]
    # except:
    #    return []


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
    # deleted fields
    if 'is_public' in d['init_data'].keys():
        d['init_data'].pop('is_public')
    if 'is_uploaded' in d['init_data'].keys():
        d['init_data'].pop('is_uploaded')

    # upward compatibility for prob_data
    if 'prob_data' not in d.keys():
        prob_keys = ['problem_type', 'y_col', 'metric', 'other_metrics', 'val_col', 'cv_folds', 'val_col_shuffle',
                     'sampling', 'holdout_ratio', 'col_submit']
        d['prob_data'] = {}
        for key in prob_keys:
            if key in d['init_data'].keys():
                d['prob_data'][key] = d['init_data'][key]
                d['init_data'].pop(key)

    # upward compatibility for calc_data
    if 'calc_data' not in d.keys():
        calc_keys = ['n_cat_cols', 'n_missing', 'x_cols', 'text_cols', 'textref_cols', 'cat_cols', 'missing_cols',
                     'is_y_categorical', 'best_is_min', 'y_n_classes', 'y_class_names']
        d['calc_data'] = {}
        for key in calc_keys:
            if key in d['load_data'].keys():
                d['calc_data'][key] = d['load_data'][key]
                d['load_data'].pop(key)
            elif key in d['init_data'].keys():
                d['calc_data'][key] = d['init_data'][key]
                d['init_data'].pop(key)

    if 'text_cols' not in d['calc_data'].keys():
        d['calc_data']['text_cols'] = []
    if 'textref_cols' not in d['calc_data'].keys():
        d['calc_data']['textref_cols'] = []
    if 'sampling' not in d['prob_data'].keys():
        d['prob_data']['sampling'] = False

    # then load dataset object
    dt = DataSet(**d['init_data'])
    dt.load(d['load_data'], d['features'])
    dt.status = get_key_store('dataset:%s:status' % dataset_id)
    # if dt.status != 'created':
    dt.update_problem(**d['prob_data'])
    dt.load_calc(d['calc_data'])

    # add counters and results
    dt.grapher = get_key_store('dataset:%s:grapher' % dataset_id)
    dt.round_counter = get_counter_store('dataset:%s:round_counter' % dataset_id)

    if include_results:
        dt.results = get_key_store('dataset:%s:results' % dataset_id)

    return dt


def create_dataset(name, domain, description, source, mode, filename_train, filename_test='', filename_cols='',
                   filename_submit='', url=''):
    """
    creates a dataset

    :param name: name of the dataset
    :param domain: domain classification of the dataset (string separated by /)
    :param description: description of the dataset
    :param source: source of the dataset
    :param mode: standard (train set), benchmark (train + test set), competition (train + submit set)
    :param filename_train: file path of the training set
    :param filename_test: name of the test set (benchmark mode)
    :param filename_cols: file to describe columns
    :param filename_submit: name of the submit set (competition mode)
    :param url: url of the dataset
    :return: dataset object
    """

    # create object and control data
    creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # others = [m for m in other_metrics.replace(' ', '').split(',') if m != '']
    dt = DataSet(0, name, domain, description, source, mode, filename_train,
                 filename_test=filename_test, filename_cols=filename_cols,
                 filename_submit=filename_submit,
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


def update_feature_dataset(dataset_id, name, description, to_keep, col_type, text_ref):
    """
    update specifically some attributes of one column

    :param dataset_id: id of the dataset
    :param name: name of the column
    :param description: new description of the column
    :param to_keep: keep this column
    :param col_type: type of the column
    :param text_ref: text set reference of the column (if col type = 'text')
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
            if f.col_type == 'text':
                f.text_ref = text_ref
            else:
                f.text_ref = ''

    # regenerate list of X columns, categoricals and text columns
    dt.update_calc()
    # then save dataset data
    dt.save(dataset_id)


def update_problem_dataset(dataset_id, problem_type, y_col, metric, other_metrics, val_col, cv_folds, val_col_shuffle,
                           sampling, holdout_ratio, col_submit):
    """
    updates the problem type of a dataset

    :param dataset_id: id of the dataset
    :param problem_type: 'regression' or 'classification'
    :param y_col: name of the target column
    :param metric: metric to be used to select the best models ('mse', 'rmse', 'log_loss', ...)
    :param other_metrics: list of secondary metrics (as a list)
    :param val_col: column name to perform the cross validation (default = 'index')
    :param cv_folds: number of cross validation folds (default = 5)
    :param val_col_shuffle: need to shuffle in cross validation (default = false)
    :param sampling: use re-sampling pre-processing (default = false)
    :param holdout_ratio: holdout ration to split train / eval set
    :param col_submit: column to use in the submit file
    """
    dt = get_dataset(dataset_id)
    dt.update_problem(problem_type, y_col, metric, other_metrics, val_col, cv_folds, val_col_shuffle,
                      sampling, holdout_ratio, col_submit)
    dt.update_y_data()
    dt.update_calc()
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


class DataSet(object):
    """
    a dataset is an object managing all the features and data of an experiment
    """

    def __init__(self, dataset_id, name, domain, description, source, mode, filename_train,
                 filename_test, filename_cols, filename_submit, url, creation_date):
        """
        creates a new dataset: it will be automatically stored

        :param name: name of the dataset
        :param domain: domain classification of the dataset (string separated by /)
        :param description: description of the dataset
        :param source: source of the dataset
        :param mode: standard (train set), benchmark (train + test set), competition (train + submit set)
        :param filename_train: file path of the training set
        :param filename_test: name of the test set
        :param filename_cols: (not implemented)
        :param url: url of the dataset
        """
        # descriptive data:
        self.dataset_id = dataset_id
        self.name = name
        self.domain = domain
        self.description = description

        self.with_test_set = False
        self.url = url
        self.source = source  # url or file id
        if filename_train == '':
            raise ValueError('filename train cannot be empty')
        # check // mode
        self.mode = mode
        if mode not in ['standard', 'benchmark', 'competition']:
            raise ValueError('mode must be standard, benchmark or competition')
        if mode == 'standard':
            if filename_test != '':
                raise ValueError('test set should be empty in standard mode')
            if filename_submit != '':
                raise ValueError('submit set should be empty in standard mode')
        elif mode == 'benchmark':
            if filename_test == '':
                raise ValueError('test set cannot be empty in benchmark mode')
            if filename_submit != '':
                raise ValueError('submit set should be empty in benchmark mode')
        else:
            # competition mode
            if filename_test != '':
                raise ValueError('test set cannot be empty in competition mode')
            if filename_submit == '':
                raise ValueError('submit set cannot be empty in competition mode')

        self.filename_train = filename_train
        self.filename_test = filename_test
        self.filename_cols = filename_cols
        self.filename_submit = filename_submit
        self.creation_date = creation_date

        # initialize other data
        self.problem_type = 'classification'
        self.y_col = ''
        self.metric = 'log_loss'
        self.other_metrics = ''
        self.val_col = ''
        self.cv_folds = 5
        self.val_col_shuffle = True
        self.sampling = False
        self.holdout_ratio = 0.2
        self.col_submit = ''
        self.n_cat_cols = 0
        self.n_missing = 0
        self.x_cols = []
        self.text_cols = []
        self.cat_cols = []
        self.missing_cols = []
        self.is_y_categorical = False
        self.best_is_min = True
        self.y_n_classes = 0
        self.y_class_names = []

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
        self.features = self.__initialize_features(df_train, df_cols)

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

        return df_cols, df_train, df_test, df_submit

    def update_stats(self, df_train, df_test, df_submit):
        # create stats on the dataset
        self.size = int(df_train.memory_usage().sum() / 1000000)
        self.n_rows = int(len(df_train) / 1000)
        self.n_cols = len(df_train.columns)
        self.test_size = int(df_test.memory_usage().sum() / 1000000)
        self.submit_size = int(df_submit.memory_usage().sum() / 1000000)

    def update_calc(self):
        # update calculated indicators on columns
        self.x_cols = [col.name for col in self.features if
                       col.to_keep and (col.name not in [self.y_col, self.val_col])]
        self.cat_cols = [col.name for col in self.features if
                         (col.name in self.x_cols) and (col.col_type == 'categorical')]
        self.text_cols = [col.name for col in self.features if (col.name in self.x_cols) and (col.col_type == 'text')]
        self.missing_cols = [col.name for col in self.features if col.n_missing > 0]

        self.n_missing = len(self.missing_cols)
        self.n_cat_cols = len(self.cat_cols)

    def update_problem(self, problem_type, y_col, metric, other_metrics, val_col, cv_folds, val_col_shuffle, sampling,
                       holdout_ratio, col_submit, best_is_min=True):
        # update problem data
        """
        creates a new dataset: it will be automatically stored

        :param problem_type: 'regression' or 'classification'
        :param y_col: name of the target column
        :param metric: metric to be used to select the best models ('mse', 'rmse', 'log_loss', ...)
        :param other_metrics: list of secondary metrics (as a list)
        :param val_col: column name to perform the cross validation (default = 'index')
        :param cv_folds: number of cross validation folds (default = 5)
        :param val_col_shuffle: need to shuffle in cross validation (default = false)
        :param sampling: use re-sampling pre-processing (default = false)
        :param holdout_ratio: holdout ration to split train / eval set
        :param col_submit: column to use in the submit file
        :param best_is_min: best is min flag required is metric == 'specific' (else we use from metric definition)
        """
        # problem and optimisation
        if problem_type not in ['regression', 'classification']:
            raise ValueError('problem type must be regression or classification')
        self.problem_type = problem_type

        self.y_col = y_col

        if self.mode != 'competition' and col_submit != '':
            raise ValueError('submit column can only be used in competition mode')
        self.col_submit = col_submit

        if metric == 'specific':
            self.best_is_min = best_is_min
        else:
            if metric not in metric_map.keys():
                raise ValueError('metric %s is not known' % metric)
            if metric_map[metric].problem_type != self.problem_type:
                raise ValueError('metric %s is not compatible with %s' % (metric, problem_type))
            self.best_is_min = metric_map[metric].best_is_min
        self.metric = metric

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

        if self.y_col != '':
            y_feature = [f for f in self.features if f.name == self.y_col][0]
            if (self.problem_type == 'regression') and (y_feature.col_type == 'categorical'):
                raise ValueError('target column %s must be numerical in regression' % self.y_col)

    def update_y_data(self):
        # update calculated data related to the y column
        y_feature = [f for f in self.features if f.name == self.y_col][0]
        self.is_y_categorical = (y_feature.col_type == 'categorical')
        self.y_n_classes = int(y_feature.n_unique_values)
        if self.problem_type == 'classification':
            uniques = self.get_data()[self.y_col].unique()
            uniques_ = [x if x == x else '' for x in uniques]
            self.y_class_names = [str(x) for x in np.sort(uniques_)]
        else:
            self.y_class_names = []

    def save(self, dataset_id):
        # saves dataset data in a pickle store
        self.dataset_id = dataset_id

        # save as json
        store = {'init_data': {'dataset_id': self.dataset_id, 'name': self.name, 'domain': self.domain,
                               'description': self.description,
                               'source': self.source, 'mode': self.mode,
                               'filename_train': self.filename_train, 'filename_test': self.filename_test,
                               'filename_cols': self.filename_cols, 'url': self.url,
                               'filename_submit': self.filename_submit, 'creation_date': self.creation_date},
                 'load_data': {'size': self.size, 'n_rows': self.n_rows, 'n_cols': self.n_cols,
                               'with_test_set': self.with_test_set},
                 'prob_data': {'problem_type': self.problem_type, 'y_col': self.y_col,
                               'metric': self.metric, 'other_metrics': self.other_metrics,
                               'val_col': self.val_col, 'cv_folds': self.cv_folds,
                               'val_col_shuffle': self.val_col_shuffle, 'sampling': self.sampling,
                               'holdout_ratio': self.holdout_ratio, 'col_submit': self.col_submit},
                 'calc_data': {'n_cat_cols': self.n_cat_cols, 'n_missing': self.n_missing,
                               'x_cols': self.x_cols, 'text_cols': self.text_cols,
                               'cat_cols': self.cat_cols, 'missing_cols': self.missing_cols,
                               'is_y_categorical': self.is_y_categorical, 'best_is_min': self.best_is_min,
                               'y_n_classes': self.y_n_classes, 'y_class_names': self.y_class_names},
                 'features': [{'name': f.name, 'raw_type': str(f.raw_type), 'n_missing': int(f.n_missing),
                               'n_unique_values': int(f.n_unique_values), 'first_unique_values': f.first_unique_values,
                               'description': f.description, 'to_keep': f.to_keep,
                               'col_type': f.col_type, 'text_ref': f.text_ref}
                              for f in self.features]
                 }
        set_key_store('dataset:%s' % self.dataset_id, store)

    def load(self, load_data, features):
        # reload data from json
        for k in load_data.keys():
            setattr(self, k, load_data[k])
        self.features = [Feature(**f) for f in features]

    def load_calc(self, calc_data):
        # reload calculated data from json
        for k in calc_data.keys():
            setattr(self, k, calc_data[k])

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
            text_ref = ''
            if col in cols:
                k = cols[col]
                description = k['description']
                if k['to_keep'] != '':
                    to_keep = k['to_keep']
                if k['col_type'] != '':
                    col_type = k['col_type']
                if 'text_ref' in k.keys() and k['text_ref'] != '':
                    text_ref = k['text_ref']

            feature = Feature(col, raw_type, n_missing, n_unique, first_unique_values, description, to_keep, col_type,
                              text_ref, n_unique_ratio)
            features.append(feature)

        return features

    def update_features(self, df):
        # updates the list of features with new dataframe

        # reset columns
        map_old_features = {f.name: f for f in self.features}
        features = []
        # retrieve column info from dataframe data
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            uniques = df[col].unique()
            n_unique = len(uniques)
            n_unique_ratio = n_unique / len(df)
            raw_type = str(df[col].dtype)
            first_unique_values = ', '.join([str(x) for x in uniques[:5]])
            if col in map_old_features:
                f = map_old_features[col]
                to_keep = f.to_keep
                description = f.description
                col_type = f.col_type
                text_ref = f.text_ref
            else:
                to_keep = True
                description = '[by feature engineering]'
                col_type = ''
                text_ref = ''

            feature = Feature(col, raw_type, n_missing, n_unique, first_unique_values, description, to_keep, col_type,
                              text_ref, n_unique_ratio)
            features.append(feature)

        self.features = features

    def __import_data(self, filename, part):
        # copy file in the dataset
        df = self.__read_data(filename)
        # save as pickle
        df.to_pickle(self.__folder() + '/data/%s.pkl' % part)


class Feature(object):
    def __init__(self, name, raw_type, n_missing, n_unique_values, first_unique_values, description, to_keep,
                 col_type, text_ref='', n_unique_ratio=0.):
        # descriptive data
        self.name = name
        self.description = description
        self.to_keep = to_keep
        self.raw_type = raw_type
        self.text_ref = text_ref

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
            if self.col_type == 'numerical' and self.n_unique_values < 10:
                self.col_type = 'categorical'
            if self.col_type == 'categorical' and self.n_unique_ratio > 0.5:
                self.col_type = 'text'
            if text_ref != '':
                if col_type != 'text':
                    raise ValueError(
                        'feature %s of type %s : text reference can only apply to text' % (name, col_type))
                if text_ref not in get_textset_list():
                    raise ValueError(
                        'feature %s of type text: text reference id %s not found' % (name, str(text_ref)))


def get_y_eval(dataset_id):
    """
    retrieves the y value of the eval set

    :param dataset_id: id of the dataset
    :return: y values
    """
    #
    return pickle.load(open(get_dataset_folder(dataset_id) + '/y_eval.pkl', 'rb'))
