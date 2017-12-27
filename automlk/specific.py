import os
import pandas as pd
from .store import exists_key_store, get_key_store, set_key_store, del_key_store
from .dataset import get_dataset_folder, get_dataset


def update_feature_engineering(dataset_id, content):
    """
    update the feature engineering content of a dataset

    :param dataset_id: id of the dataset
    :param content: text of the python function (without def and return)
    :return:
    """

    with open(get_dataset_folder(dataset_id) + '/data/fe_%s.txt' % dataset_id, 'w') as f:
        f.write(content)
    # then update df X and features
    dt = get_dataset(dataset_id)
    X = dt.get_data()
    # apply specific engineering to X
    X = apply_feature_engineering(dataset_id, X)
    dt.update_features(X)
    dt.save(dataset_id)


def delete_feature_engineering(dataset_id):
    """
    delete the feature engineering function

    :param dataset_id: id of the dataset
    :return:
    """
    filename = get_dataset_folder(dataset_id) + '/data/fe_%s.txt' % dataset_id
    if os.path.exists(filename):
        os.remove(filename)


def get_feature_engineering(dataset_id):
    """
    retrieves the feature engineering content of a dataset

    :param dataset_id: id of the dataset
    :return: content: text of the python function (without def and return)
    """
    filename = get_dataset_folder(dataset_id) + '/data/fe_%s.txt' % dataset_id
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return "".join(f.readlines())
    else:
        return ''


def apply_feature_engineering(dataset_id, X=None):
    """
    executes the feature engineering definition

    :param dataset_id: dataset id
    :return:
    """
    global specific_feature_engineering
    exec_feature_engineering(get_feature_engineering(dataset_id))
    if isinstance(X, pd.DataFrame):
        return specific_feature_engineering(X)
    else:
        return None


def exec_feature_engineering(content):
    """
    executes the feature engineering definition from content

    :param content: content of the function
    :return:
    """
    global specific_feature_engineering
    # prepare specific function definition
    if content == '':
        return
    s_indent = " " * 4
    f = 'def specific_feature_engineering(X):\n' + content + '\nreturn X'
    f = f.replace('\n', '\n' + s_indent).replace('\t', s_indent)
    # execute function
    exec(f, globals())


def update_specific_metrics(dataset_id, name, best_is_min, content):
    """
    update the specific metrics content of a dataset

    :param dataset_id: id of the dataset
    :param name: name of the metrics
    :param content: text of the python function (without def)
    :return:
    """
    set_key_store('dataset:%s:metrics' % dataset_id, {'name': name, 'best_is_mean': best_is_min})
    with open(get_dataset_folder(dataset_id) + '/data/sm_%s.txt' % dataset_id, 'w') as f:
        f.write(content)


def delete_specific_metrics(dataset_id):
    """
    delete the specific metrics content of a dataset

    :param dataset_id: id of the dataset
    :return:
    """
    key = 'dataset:%s:metrics' % dataset_id
    if exists_key_store(key):
        del_key_store(key)


def get_specific_metrics(dataset_id):
    """
    retrieves the specific metrics content of a dataset

    :param dataset_id: id of the dataset
    :return: specific, name, content (text of the python function (without def))
    """
    key = 'dataset:%s:metrics' % dataset_id
    content = ''
    best_is_min = True
    if exists_key_store(key):
        m = get_key_store(key)
        name = m['name']
        if best_is_min in m.keys():
            best_is_min = m['best_is_min']
        filename = get_dataset_folder(dataset_id) + '/data/sm_%s.txt' % dataset_id
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = "".join(f.readlines())
    else:
        name = ''
    return name, best_is_min, content


def apply_specific_metrics(dataset_id):
    """
    executes the specific_metrics definition

    :param dataset_id: dataset id
    :return:
    """
    global specific_metrics
    name, best_is_min, content = get_specific_metrics(dataset_id)
    exec_specific_metrics(content)


def return_specific_metrics(y_act, y_pred):
    """
    return the value of the metrics

    :param dataset_id: dataset id
    :return:
    """
    global specific_metrics
    return specific_metrics(y_act, y_pred)


def exec_specific_metrics(content):
    """
    executes the specific_metrics definition from content

    :param content: content of the function
    :return:
    """
    global specific_metrics
    # prepare specific function definition
    if content == '':
        return
    s_indent = " " * 4
    f = 'def specific_metrics(y_act, y_pred):\n' + content
    f = f.replace('\n', '\n' + s_indent).replace('\t', s_indent)
    # execute function
    exec(f, globals())
