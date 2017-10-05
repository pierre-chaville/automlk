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
    if os.path.exists('../config.json'):
        with open('../config.json', 'r') as f:
            return eval("".join(f.readlines()))
    raise EnvironmentError('configuration file %s not found' % '../config.json')


def get_data_folder():
    """
    retrieves root folder from 'automlk.json' configuration file
    :return: storage folder of the data
    """
    return get_config()['data']

class HyperContext():

    def __init__(self, problem_type, feature_names, cat_cols, missing_cols):
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.missing_cols = missing_cols.copy()