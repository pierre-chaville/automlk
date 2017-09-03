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
        data_dir = home + '/.automlk'
    else:
        data_dir = home + '/automlk'

    if not os.path.exists(data_dir + '/automlk.json'):
        print('creating root folder and configuration file')

        # create default folder
        os.makedirs(data_dir)

        # create default configuration file
        with open(data_dir + '/automlk.json', 'w') as f:
            f.write('{"data": "%s", "theme": "darkly"}\n' % data_dir)
        return data_dir
    else:
        with open(data_dir + '/automlk.json', 'r') as f:
            config = eval("".join(f.readlines()))
            return config['data']


class HyperContext():

    def __init__(self, problem_type, feature_names, cat_cols, missing_cols):
        self.problem_type = problem_type
        self.pipeline = []
        self.feature_names = feature_names.copy()
        self.cat_cols = cat_cols.copy()
        self.missing_cols = missing_cols.copy()