import pickle, os, glob
import pandas as pd
import numpy as np
import datetime
from .metrics import metric_map
from .context import METRIC_NULL, get_dataset_folder, get_data_folder
from .graphs import graph_correl_features, graph_histogram


class DataSet(object):
    """
    class to describe a dataset
    """

    def __init__(self, name, description, problem_type, y_col, is_uploaded, source, filename_train, metric,
                 other_metrics=[], val_col='index', cv_folds=5, val_col_shuffle=True,
                 holdout_ratio=0.2, filename_test='', filename_cols='', is_public=False, url=''):

        # TODO: add feature details from file or dataframe

        # descriptive data:
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

        if metric not in metric_map.keys():
            raise ValueError('metric %s is not known' % metric)
        if metric_map[metric][1] != self.problem_type:
            raise ValueError('metric %s is not compatible with %s' % (metric, problem_type))
        self.metric = metric

        self.best_is_min = metric_map[metric][2]

        for m in other_metrics:
            if m not in metric_map.keys():
                raise ValueError('other metric %s is not known' % m)
            if metric_map[m][1] != self.problem_type:
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

        # check train data
        self.__check_data(filename_train)
        self.filename_train = filename_train
        df_train = self.__read_data(filename_train)
        if y_col not in df_train.columns:
            raise ValueError('y_col %s not in the columns of the dataset' % y_col)
        self.y_col = y_col

        self.features, self.is_y_categorical, self.y_n_classes, self.y_class_names = self.__initialize_features(
            df_train)

        # check test data
        self.filename_test = filename_test
        if filename_test != '':
            self.__check_data(filename_test)
            df_test = self.__read_data(filename_test)
            self.with_test_set = True
            self.holdout_ratio = 0

        self.x_cols = [col.name for col in self.features if
                       col.to_keep and (col.name not in [self.y_col, self.val_col])]

        self.cat_cols = [col.name for col in self.features if
                         (col.name in self.x_cols) and (col.col_type == 'categorical')]

        self.missing_cols = [col.name for col in self.features if col.n_missing > 0]

        # create stats on the dataset
        self.size = int(df_train.memory_usage().sum() / 1000000)
        self.n_rows = int(len(df_train) / 1000)
        self.n_cols = len(df_train.columns)
        self.n_cat_cols = len(self.cat_cols)
        self.n_missing = len(self.missing_cols)

        self.creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__save()
        self.__create_graphs(df_train, 'train')
        if filename_test != '':
            self.__create_graphs(df_test, 'test')

    def __folder(self):
        # storage folder of the dataset
        return get_dataset_folder(self.uid)

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

    def get_data(self, part='train'):
        # returns the imported data of the dataset as a dataframe (after __import_data)
        return pd.read_pickle(self.__folder() + '/data/%s.pkl' % part)

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
            raw_type = df[col].dtype
            feature = Feature(col, raw_type, n_missing, n_unique, uniques)
            features.append(feature)

            # y_col : categorical if classification, numerical if regression
            if col == self.y_col:
                if (self.problem_type == 'regression') and (feature.col_type == 'categorical'):
                    raise ValueError('target column %s must be numerical in regression' % col.name)

                is_y_categorical = (feature.col_type == 'categorical')
                y_n_classes = feature.n_unique_values
                y_class_names = np.sort(uniques)

        return features, is_y_categorical, y_n_classes, y_class_names

    def evaluate_metric(self, y_act, y_pred, metric=None):
        # evaluate score with the metric of the dataset
        if not metric:
            metric = self.metric
        else:
            if metric not in self.other_metrics:
                raise ValueError('evaluation metric not listed in other metrics')
        function = metric_map[metric][0]
        best_is_min = metric_map[metric][2]
        try:
            # use sign before metrics to always compare best is min in comparisons
            # but requires to display abs value for display
            if best_is_min:
                return function(y_act, y_pred)
            else:
                return -function(y_act, y_pred)
        except Exception as e:
            print('error in evaluating metric %s: %s' % (metric, e))
            return METRIC_NULL

    def __save(self):
        # saves dataset data in a pickle store
        self.__create_uid()
        self.__create_folders()
        self.__import_data(self.filename_train, 'train')
        if self.filename_test != '':
            self.__import_data(self.filename_test, 'test')
        pickle.dump(self, open(self.__folder() + '/dataset.pkl', 'wb'))

    def __create_uid(self):
        # find id for the dataset
        for i in range(1, 1000000):
            if not os.path.isdir(get_data_folder() + '/%d' % i):
                self.uid = str(i)
                return

    def __create_folders(self):
        # create folders
        root = get_data_folder() + '/%s' % self.uid
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
        graph_histogram(self.uid, self.y_col, self.is_y_categorical, df[self.y_col].values, part)

        if part == 'train':
            # histogram of the target columns
            graph_correl_features(self, df)

            # histograpm for each feature
            """
            for col in self.x_cols:
                graph_histogram(self.uid, col, df[col].values, part)
            """

    # TODO: implement update dataset
    # TODO: implement delete dataset
    """
    def delete(self):
        # deletes the descriptive data of a dataset and the uploaded data
        os.remove(dataset_filename(self.uid))
    """

class Feature(object):
    def __init__(self, name, raw_type, n_missing, n_unique_values, unique_values):
        # descriptive data
        self.name = name
        self.description = ''
        self.to_keep = True
        self.raw_type = raw_type

        # initialize type
        if raw_type == np.float:
            self.col_type = 'numerical'
        elif raw_type == np.int:
            self.col_type = 'numerical'
        elif raw_type in [str, object]:
            self.col_type = 'categorical'
        # TODO : manage dates and text

        # calculated data
        self.n_missing = n_missing
        self.n_unique_values = n_unique_values
        self.first_unique_values = unique_values[:5]
        self.print_values = ', '.join([str(x) for x in self.first_unique_values])


def get_dataset_list():
    """
    get the list of all datasets
    :return: list of datasets objects
    """
    # returns the list of datasets ids
    dl = []
    for file in glob.glob(get_data_folder() + '/*/'):
        uid = file.split('/')[-2]
        dl.append(get_dataset(uid))
    return dl


def get_dataset(dataset_uid):
    """
    get the descriptive data of a dataset
    :param dataset_uid: id of the dataset
    :return: dataset object
    """
    return pickle.load(open(get_dataset_folder(dataset_uid) + '/dataset.pkl', 'rb'))


