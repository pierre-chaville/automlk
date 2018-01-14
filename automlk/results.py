import pickle
import pandas as pd
import numpy as np
from .store import exists_key_store, get_key_store
from .dataset import get_dataset_list, get_dataset_folder, get_dataset
from .prepare import get_idx_train_test, get_eval_sets


def get_importance(dataset_id, round_id):
    """
    features importance of the model

    :param dataset_id: id of the dataset
    :param round_id: id of the round
    :return: feature importance as a dataframe
    """
    try:
        return pickle.load(open(get_dataset_folder(dataset_id) + '/features/%s.pkl' % round_id, 'rb'))
    except:
        return None


def get_pred_eval_test(dataset_id, round_id):
    """
    prediction on eval set & test & submit set

    :param dataset_id: id of the dataset
    :param round_id: id of the round
    :return: list of predictions for eval set, test and submit set
    """
    return pickle.load(open(get_dataset_folder(dataset_id) + '/predict/%s.pkl' % round_id, 'rb'))


def print_value(x):
    # easy print function for dictionary value
    return ('%6.4f' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)


def get_home_best():
    # get the list of datasets with their best results
    datasets = get_dataset_list(include_results=True)[::-1]
    for dt in datasets:
        if dt.status != 'created':
            best = get_best_models(dt.dataset_id)
            if len(best) > 0:
                best = best[0]
                dt.best_round_id = best['round_id']
                dt.best_model_name = best['model_name']
                dt.best_score_eval = best['score_eval']
                dt.best_score_test = best['score_test']
                dt.best_cv_mean = best['cv_mean']
                dt.best_cv_std = best['cv_std']
                dt.best_cv_max = best['cv_max']

    return datasets


def get_best_models(dataset_id):
    # get the best results per model
    key = 'dataset:%s:best' % dataset_id
    if exists_key_store(key):
        return get_key_store(key)
    else:
        return []


def get_best_pp(dataset_id):
        # get the best results per pre-processing
        key = 'dataset:%s:best_pp' % dataset_id
        if exists_key_store(key):
            return get_key_store(key)
        else:
            return []


def get_best_details(df, model_name):
    # get the best results for a model
    best = df[df.model_name == model_name].sort_values(by='cv_max')

    # create params detailed dataframe
    params = []
    for p, round_id in zip(best.model_params.values, best.round_id.values):
        params.append({**{'round_id': round_id}, **p})

    params = pd.DataFrame(params)
    if len(params) > 1:
        to_drop = []
        # remove cols with 1 unique value
        for col in params.columns:
            l = params[col].map(str).unique()
            if len(l) <= 1:
                to_drop.append(col)
        if len(to_drop) > 0:
            params.drop(to_drop, axis=1, inplace=True)

    # strip underscores in column names
    new_col = []
    for col in params.columns:
        if col != 'round_id':
            new_col.append(col.replace('_', ' '))
        else:
            new_col.append(col)
    params.columns = new_col

    # round floating values
    for col in params.columns:
        if col != 'round_id':
            params[col] = params[col].fillna('').map(print_value)

    # params.fillna('', inplace=True)
    best = pd.merge(best, params, on='round_id')

    # relative performance
    best['rel_score'] = abs(100 * (best['cv_max'] - best['cv_max'].max()) / (best['cv_max'].max() - best['cv_max'].min()))
    return [col for col in params.columns if col != 'round_id'], best


def __select_process(process, pipeline):
    # select the element in the pipeline with category c
    for p in pipeline:
        if p[2] == process:
            return p
    return '', '', '', ''


def get_best_details_pp(df, process_name):
    # get the best results for a model
    df['is_selected'] = df.pipeline.map(lambda x: __select_process(process_name, x)[2] != '')
    df = df[df.is_selected]

    df['cat_ref'] = df['pipeline'].map(lambda x: __select_process(process_name, x)[0])
    df['cat_name'] = df['pipeline'].map(lambda x: __select_process(process_name, x)[1])
    df['cat_process'] = df['pipeline'].map(lambda x: __select_process(process_name, x)[2])
    df['cat_params'] = df['pipeline'].map(lambda x: __select_process(process_name, x)[3])

    best = df.sort_values(by='cv_max')

    # create params detailed dataframe
    params = []
    for p, round_id in zip(best.cat_params.values, best.round_id.values):
        params.append({**{'round_id': round_id}, **p})

    params = pd.DataFrame(params)
    if len(params) > 1:
        to_drop = []
        # remove cols with 1 unique value
        for col in params.columns:
            l = params[col].map(str).unique()
            if len(l) <= 1:
                to_drop.append(col)
        if len(to_drop) > 0:
            params.drop(to_drop, axis=1, inplace=True)

    # strip underscores in column names
    new_col = []
    for col in params.columns:
        if col != 'round_id':
            new_col.append(col.replace('_', ' '))
        else:
            new_col.append(col)
    params.columns = new_col

    # round floating values
    for col in params.columns:
        if col != 'round_id':
            params[col] = params[col].fillna('').map(print_value)

    # params.fillna('', inplace=True)
    best = pd.merge(best, params, on='round_id')

    # relative performance
    best['rel_score'] = abs(100 * (best['cv_max'] - best['cv_max'].max()) / (best['cv_max'].max() - best['cv_max'].min()))
    return [col for col in params.columns if col != 'round_id'], best


def get_data_steps(process):
    # generate a list of process steps from the json description
    steps = []
    if isinstance(process, dict):
        for step_name in process.keys():
            params = process[step_name]
            steps.append((step_name, [(key, params[key]) for key in params.keys()]))
    return steps


def get_feature_steps(process_name, params):
    # generate a list of process steps from the json description
    return (process_name, [(key, params[key]) for key in params.keys()])


def get_round_params(df, round_id):
    # details for a round
    round = df[df.round_id == int(round_id)]
    params = round.model_params.values[0].copy()
    for key in params.keys():
        params[key] = print_value(params[key])
    return params


def get_feature_importance(dataset_id, round_id):
    # get feature importance for the selected model round

    df = get_importance(dataset_id, round_id)
    if not isinstance(df, pd.DataFrame) or 'importance' not in df.columns:
        return []

    df['pct_importance'] = np.round(100 * df.importance / df.importance.sum(), 1)
    df['rel_importance'] = np.round(100 * df.importance / df.importance.max(), 1)

    return df.sort_values('importance', ascending=False).to_dict(orient='records')


def create_predict_file(dataset_id, round_id):
    """
    generate a prediction file in excel format on eval set, with original data and prediction

    :param dataset_id: dataset id
    :param round_id: round id
    :return: None
    """
    dataset = get_dataset(dataset_id)

    # load original train data and indexes
    df = dataset.get_data()
    i_train, i_test = get_idx_train_test(dataset_id)
    df = df.iloc[i_train]

    ds = get_eval_sets(dataset_id)

    # load prediction results
    y_pred_eval, y_pred_test, y_pred_submit = get_pred_eval_test(dataset_id, round_id)

    cols = dataset.x_cols
    if dataset.problem_type == 'regression':
        df['_predict'] = y_pred_eval
        df['_actual'] = ds.y_train
        df['_delta'] = df['_actual'] - df['_predict']
        df = df[['_predict', '_actual', '_delta'] + cols]
    else:
        df['_predict'] = np.argmax(y_pred_eval, axis=1)
        df['_p_class'] = df['_predict'].map(lambda x: dataset.y_class_names[x])
        df['_actual'] = ds.y_train
        df['_a_class'] = df['_actual'].map(lambda x: dataset.y_class_names[x])
        df['_delta'] = df['_actual'] != df['_predict']
        df = df[['_predict', '_actual', '_delta', '_p_class', '_a_class'] + cols]

    # save as excel file
    df.to_excel(get_dataset_folder(dataset_id) + '/submit/predict_%s.xlsx' % round_id, index=False)

