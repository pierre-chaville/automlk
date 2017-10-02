#coding: utf-8

import os, platform
from pathlib import Path

METRIC_NULL = 1e7
SEARCH_QUEUE = 'controller:search_queue'
RESULTS_QUEUE = 'controller:results_queue'
CONTROLLER_ID = 'controller:dataset_id'


def get_dataset_folder(dataset_uid):
    """
    storage folder of the dataset
    :param dataset_uid: id of the dataset
    :return: folder path (root of the structure)
    """
    return get_data_folder() + '/%s' % dataset_uid


def get_config():
    """

    retrieves configuration parameters
    :return: config dict
    """
    """
    # detect OS
    home = str(Path.home())
    if platform.system() == 'Linux':
        setup_dir = home + '/.automlk'
    else:
        setup_dir = home.replace('\\', '/') + '/automlk'

    if os.path.exists(setup_dir + '/automlk.json'):
        with open(setup_dir + '/automlk.json', 'r') as f:
            return eval("".join(f.readlines()))
    raise EnvironmentError('configuration file %s not found' % setup_dir + '/automlk.json')
    """
    if os.path.exists('../config.json'):
        with open('../config.json', 'r') as f:
            return eval("".join(f.readlines()))
    raise EnvironmentError('configuration file %s not found' % '../config.json')


def get_data_folder():
    """
    retrieves root folder from 'automlk.json' configuration file
    :return: storage folder of the data
    """

    """
    # detect OS
    home = str(Path.home())
    if platform.system() == 'Linux':
        setup_dir = home + '/.automlk'
    else:
        setup_dir = home.replace('\\', '/') + '/automlk'

    if not os.path.exists(setup_dir + '/automlk.json'):
        print('creating setup folder and configuration file')

        # create setup and store folders
        os.makedirs(setup_dir)
        os.makedirs(setup_dir + '/store')

        # create default configuration file
        with open(setup_dir + '/automlk.json', 'w') as f:
            f.write('{"data": "%s", "theme": "darkly", "store": "localhost"}\n' % setup_dir)
        return setup_dir
    else:
        # read the data folder in the setup file
        # can be modified by user in order to manage distributed workers on different machines in parallel
        with open(setup_dir + '/automlk.json', 'r') as f:
            config = eval("".join(f.readlines()))
            if platform.system() == 'Linux':
                return config['data']
            else:
                return os.path.abspath(config['data']).replace('\\', '/') + '/automlk'
    """
    return get_config()['data']

class HyperContext():

    def __init__(self, problem_type, feature_names, cat_cols, missing_cols):
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.missing_cols = missing_cols.copy()