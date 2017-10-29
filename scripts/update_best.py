import pandas as pd
from automlk.dataset import get_dataset_list
from automlk.worker import get_search_rounds
from automlk.graphs import graph_history_search
from automlk.store import set_key_store
"""
module specifically designed to update search graphs and best models and pp
after new version (results are calculated by the controller)
"""

def __get_best_models(df):
    # get the best results per model
    if len(df) < 1:
        return pd.DataFrame()
    best = df.sort_values(by=['model_name', 'cv_max']).groupby('model_name', as_index=False).first().sort_values(
        by='cv_max').fillna('')
    counts = df[['model_name', 'round_id']].groupby('model_name', as_index=False).count()
    counts.columns = ['model_name', 'searches']
    # relative performance
    best['rel_score'] = abs(100 * (best.cv_max - best.cv_max.max()) / (best.cv_max.max() - best.cv_max.min()))
    return pd.merge(best, counts, on='model_name')


def __get_best_pp(df):
    # get the best results per pre-processing

    df = df[df.level == 1].copy()
    if len(df) < 1:
        return []

    # find categories used for this model
    cat = []
    for p in df.pipeline.values:
        cat += [s[1] for s in p if not isinstance(s, dict)]
    cat = set(cat)

    # for each category, we will want to find the best model
    all_cat = []
    for c in ['text', 'categorical', 'missing', 'scaling', 'feature']:
        if c in cat:
            df['cat_ref'] = df['pipeline'].map(lambda x: __select_cat(c, x)[0])
            df['cat_name'] = df['pipeline'].map(lambda x: __select_cat(c, x)[1])
            df['cat_process'] = df['pipeline'].map(lambda x: __select_cat(c, x)[2])
            df['cat_params'] = df['pipeline'].map(lambda x: __select_cat(c, x)[3])

            best = df.sort_values(by=['cat_ref', 'cv_max']).groupby('cat_ref', as_index=False).first().sort_values(
                by='cv_max').fillna('')

            counts = df[['cat_ref', 'round_id']].groupby('cat_ref', as_index=False).count()
            counts.columns = ['cat_ref', 'searches']

            # relative performance
            best['rel_score'] = abs(100 * (best.cv_max - best.cv_max.max()) / (best.cv_max.max() - best.cv_max.min()))

            all_cat.append((c, pd.merge(best, counts, on='cat_ref').to_dict(orient='records')))

    return all_cat


def __select_cat(c, pipeline):
    # select the element in the pipeline with category c
    for p in pipeline:
        if p[1] == c:
            return p
    return '', '', '', ''


for dt in get_dataset_list(include_status=True):
    if dt.status != 'created':
        print(dt.name)
        # get search history
        df = get_search_rounds(dt.dataset_id)

        # generate graphs
        best = __get_best_models(df)
        best1 = best[best.level == 1]
        best2 = best[best.level == 2]
        graph_history_search(dt, df, best1, 1)
        graph_history_search(dt, df, best2, 2)

        # then update best models & pp
        set_key_store('dataset:%s:best' % dt.dataset_id, best.to_dict(orient='records'))
        set_key_store('dataset:%s:best_pp' % dt.dataset_id, __get_best_pp(df))