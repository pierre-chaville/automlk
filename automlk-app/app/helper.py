import pandas as pd
import numpy as np
from automlk.search import get_importance #, get_pred_eval_test, METRIC_NULL
#from automlk.dataset import get_y_eval

def dataset_folder(dt_uid):
    # return the folder where search results are stored
    return '../datasets/search/%s' % dt_uid


def print_value(x):
    # easy print function for dictionary value
    return ('%6.4f' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)


def get_best_models(df):
    # get the best results per model

    if len(df) < 1:
        return pd.DataFrame()

    best = df.sort_values(by=['model_name', 'score_eval']).groupby('model_name', as_index=False).first().sort_values(
        by='score_eval').fillna('')

    counts = df[['model_name', 'round_id']].groupby('model_name', as_index=False).count()
    counts.columns = ['model_name', 'searches']

    # relative performance
    best['rel_score'] = abs(100 * (best.score_eval - best.score_eval.max()) / (best.score_eval.max() - best.score_eval.min()))

    return pd.merge(best, counts, on='model_name')


def get_best_details(df, model_name):
    # get the best results for a model
    best = df[df.model_name == model_name].sort_values(by='score_eval')

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
    best['rel_score'] = abs(100 * (best['score_eval'] - best['score_eval'].max()) / (best['score_eval'].max() - best['score_eval'].min()))
    return [col for col in params.columns if col != 'round_id'], best


def get_process_steps(process):
    # generate a list of process steps from the json description
    steps = []
    for step_name in process.keys():
        params = process[step_name]
        steps.append((step_name, [(key, params[key]) for key in params.keys()]))
    return steps


def get_round_params(df, round_id):
    # details for a round
    round = df[df.round_id == round_id]
    params = round.model_params.values[0].copy()
    for key in params.keys():
        params[key] = print_value(params[key])
    return params


def get_feature_importance(uid, round_id):
    # get feature importance for the selected model round

    df = get_importance(uid, round_id)
    if not isinstance(df, pd.DataFrame):
        return []

    df['pct_importance'] = np.round(100 * df.importance / df.importance.sum(), 1)
    df['rel_importance'] = np.round(100 * df.importance / df.importance.max(), 1)

    return df.sort_values('importance', ascending=False).to_dict(orient='records')


