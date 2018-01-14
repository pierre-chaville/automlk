import os
import json
from .config import set_use_redis
from .xyset import XySet


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
            if 'graph_theme' not in config.keys():
                config['graph_theme'] = 'dark'
            if 'doc_theme' not in config.keys():
                config['doc_theme'] = 'default'
            return config
    raise EnvironmentError('configuration file %s not found' % '../config.json')


def set_config(data, theme, bootstrap, graph_theme, store, store_url):
    """
    set config data

    :param data: path to data storage
    :param theme: theme for user interface
    :param bootstrap: specific url for a bootstrap
    :param graph_theme: style for graphs (dark / white)
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
    config = {'data': data, 'theme': theme, 'bootstrap': bootstrap, 'graph_theme': graph_theme, 'store': store,
              'store_url': store_url}
    with open('../config.json', 'w') as f:
        f.write(json.dumps(config) + '\n')


def text_model_filename(textset_id, model_type, params):
    """
    name of the file with params

    :param model_type: model type (bow, w2v, d2v)
    :param params: params of the model
    :return:
    """
    folder = get_data_folder() + '/texts/%s' % textset_id
    if not os.path.exists(folder):
        os.makedirs(folder)

    params_name = "-".join([key + '_' + str(params[key]) for key in params.keys()])
    for c in ['[', ']', ',', '(', ')', '{', '}']:
        params_name = params_name.replace(c, '')
    return folder + '/%s-' % model_type + params_name.replace(' ', '_')


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
