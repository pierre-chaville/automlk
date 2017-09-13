#coding: utf-8

import os, platform
from pathlib import Path

METRIC_NULL = 1e7

def get_dataset_folder(dataset_uid):
    """
    storage folder of the dataset
    :param dataset_uid: id of the dataset
    :return: folder path (root of the structure)
    """
    return get_data_folder() + '/%s' % dataset_uid


def get_data_folder():
    """
    retrieves root folder from 'automlk.json' configuration file
    :return: storage folder of the data
    """
    # detect OS
    home = str(Path.home())
    if platform.system() == 'Linux':
        setup_dir = home + '/.automlk'
    else:
        setup_dir = home.replace('\\', '/') + '/automlk'

    if not os.path.exists(setup_dir + '/automlk.json'):
        print('creating setup folder and configuration file')

        # create setup folder
        os.makedirs(setup_dir)

        # create default configuration file
        with open(setup_dir + '/automlk.json', 'w') as f:
            f.write('{"data": "%s", "theme": "darkly"}\n' % setup_dir)
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


class HyperContext():

    def __init__(self, problem_type, feature_names, cat_cols, missing_cols):
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.missing_cols = missing_cols.copy()