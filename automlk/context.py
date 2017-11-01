import os
import json
from .config import set_use_redis


class HyperContext(object):
    """
    this class stores the various data required for pre-processing
    """

    def __init__(self, problem_type, feature_names, cat_cols, text_cols, missing_cols):
        """

        :param problem_type: problem type (regresion / classification)
        :param feature_names: list of names of the features (for the context only, not the dataset feature)
        :param cat_cols: categorical columns for the context
        :param text_cols: text columns for the context
        :param missing_cols: missing columns for the context
        """
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.text_cols = text_cols.copy()
        self.missing_cols = missing_cols.copy()


class XySet(object):
    def __init__(self, X, y, X_train, y_train, X_test, y_test, X_submit,
                 id_submit, cv_folds, y_eval_list, y_eval, i_eval):
        """
        this class stores the various data required for analysis and hyepr-optimisation

        :param X: X features for the complete dataset (excluding bechnmark/test and submit)
        :param y: y for this complete set
        :param X_train: X features for the train set after holdout
        :param y_train: y for this set
        :param X_test: X features for the holdout set or test set in benchmark
        :param y_test: y for this set
        :param X_submit: X features for the submit set (competition mode only)
        :param id_submit: y for this set
        :param cv_folds: cross validation folds
        :param y_eval_list: list of folds, with indexes for each fold
        :param y_eval: y for the eval set
        :param i_eval: indexes for the eval set
        """
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_submit = X_submit
        self.id_submit = id_submit
        self.cv_folds = cv_folds
        self.y_eval_list = y_eval_list
        self.y_eval = y_eval
        self.i_eval = i_eval


def get_dataset_folder(dataset_id):
    """
    storage folder of the dataset
    :param dataset_id: id of the dataset
    :return: folder path (root of the structure)
    """
    return get_data_folder() + '/%s' % dataset_id


def get_config():
    """

    retrieves configuration parameters
    :return: config dict
    """
    if os.path.exists('../config.json'):
        with open('../config.json', 'r') as f:
            config = eval("".join(f.readlines()))
            # upward compatibility
            if 'bootstrap' not in config.keys():
                config['bootstrap'] = ''
            if 'graph_style' not in config.keys():
                config['graph_style'] = 'ggplot'
            if 'doc_theme' not in config.keys():
                config['doc_theme'] = 'default'
            return config
    raise EnvironmentError('configuration file %s not found' % '../config.json')


def set_config(data, theme, bootstrap, graph_style, store, store_url):
    """
    set config data

    :param data: path to data storage
    :param theme: theme for user interface
    :param bootstrap: specific url for a bootstrap
    :param graph_style: style for graphs (see matplotlib documentation)
    :param store: store mode (redis / file)
    :param store_url: url if redis mode
    :return:
    """
    # check data
    if not os.path.exists(data):
        raise EnvironmentError('data folder %s do not exist' % data)

    if store == 'redis':
        # check connection to redis
        try:
            import redis
            rds = redis.Redis(host=store_url)
            rds.exists('test')
        except:
            raise EnvironmentError('could not connect to redis')
        set_use_redis(True)
    else:
        store_folder = data + '/store'
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        set_use_redis(False)

    # then save config
    config = {'data': data, 'theme': theme, 'bootstrap': bootstrap, 'graph_style': graph_style, 'store': store,
              'store_url': store_url}
    with open('../config.json', 'w') as f:
        f.write(json.dumps(config) + '\n')


def get_data_folder():
    """
    retrieves root folder from 'automlk.json' configuration file
    :return: storage folder name of the data
    """
    return get_config()['data']


def get_uploads_folder():
    """
    folder to store file uploads

    :return: folder name
    """
    folder = get_data_folder() + '/uploads'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
