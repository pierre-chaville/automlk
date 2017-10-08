import os
import json
from .config import set_use_redis

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
            return eval("".join(f.readlines()))
    raise EnvironmentError('configuration file %s not found' % '../config.json')


def set_config(data, theme, store, store_url):
    """
    set config data

    :param data: path to data storage
    :param theme: theme for user interface
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
        store_folder = data+'/store'
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        set_use_redis(False)

    # then save config
    config = {'data': data, 'theme': theme, 'store': store, 'store_url': store_url}
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


class HyperContext():

    def __init__(self, problem_type, feature_names, cat_cols, missing_cols):
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.missing_cols = missing_cols.copy()