from .config import *
from .store import *
from .dataset import get_dataset_ids, get_dataset, create_dataset_sets
from .solutions import *
from .solutions_pp import *
from .worker import get_search_rounds
from .monitor import heart_beep
from .graphs import graph_history_search

PATIENCE_RANDOM = 100
PATIENCE_ENSEMBLE = 100


def launch_controller():
    """
    controller process: manages the search strategy and send instruction to workers
    :return:
    """

    i_dataset = -1
    # controls the optimization rounds and sends instructions to the workers
    while True:
        # check the list of datasets to search
        active = [id for id in get_dataset_ids() if get_key_store('dataset:%s:status' % id) == 'searching']

        if len(active) == 0:
            heart_beep('controller', {})
            time.sleep(1)
            continue

        # sends work to the workers when their queue is empty
        if llen_key_store(SEARCH_QUEUE) == 0:
            # get next dataset to search
            i_dataset += 1
            if i_dataset > len(active) - 1:
                i_dataset = 0

            # retrieves dataset and status of search
            dataset_id = active[i_dataset]

            # find search job
            msg_search = __create_search_round(dataset_id)

            # send queue the next job to do
            print('sending %s' % msg_search)
            lpush_key_store(SEARCH_QUEUE, msg_search)
            heart_beep('controller', msg_search)

        # then read the results queue
        while llen_key_store(RESULTS_QUEUE) > 0:
            msg_result = brpop_key_store(RESULTS_QUEUE)
            __process_result(msg_result)

        # then wait 1 second
        time.sleep(1)


def __create_search_round(dataset_id):
    # create a search solution

    dataset = get_dataset(dataset_id)
    search = __find_search_store(dataset)

    # generate round id
    round_id = incr_key_store('dataset:%s:round_counter' % dataset_id) - 1
    if round_id == 0:
        # first launch: create train & eval & test set
        create_dataset_sets(dataset)

    # generate model and model params
    i_round = round_id % len(search['choices'])
    ref = search['choices'][i_round]
    solution = model_solutions_map[ref]
    default_mode = False
    if search['level'] == 1:
        if round_id < len(search['choices']) - 1:
            # mode = default
            params = solution.default_params
            default_mode = True
        else:
            params = get_random_params(solution.space_params)
    else:
        search['ensemble_depth'] = random.randint(1, 10)
        params = get_random_params(solution.space_params)

    # generate pre-processing pipeline and pre-processing params
    pipeline = __get_pipeline(dataset, default_mode, round_id)

    # generate search message
    msg_search = {**search, **{'dataset_id': dataset.dataset_id, 'round_id': round_id, 'solution': solution.ref,
                               'model_name': solution.name, 'model_params': params, 'pipeline': pipeline,
                               'time_limit': __time_limit(dataset)}}
    msg_search.pop('choices')
    msg_search.pop('start')
    return msg_search


def __find_search_store(dataset):
    # add data for this dataset from the store
    if exists_key_store('dataset:%s:search' % dataset.dataset_id):
        search = get_key_store('dataset:%s:search' % dataset.dataset_id)
        return search
    else:
        return {'start': 0, 'level': 1, 'threshold': 0,
                'choices': __get_model_class_list(dataset, 1),
                }


def __time_limit(dataset):
    # determine max delay to execute search
    if dataset.n_rows < 10:
        return 120 * int(1 + dataset.n_rows)
    elif dataset.n_rows < 100:
        return 1800
    else:
        return 3600


def __get_pipeline(dataset, default_mode, i_round):
    # generates the list of potential data pre-processing depending on the problem type
    pipeline = []
    # X pre-processing: text
    if len(dataset.text_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'text', default_mode, i_round))
    # X pre-processing: categorical
    if len(dataset.cat_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'categorical', default_mode, i_round))
    # missing values
    if len(dataset.missing_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'missing', default_mode, i_round))
    # scaling
    pipeline.append(__get_pp_choice(dataset, 'scaling', default_mode, i_round))
    # feature selection
    pipeline.append(__get_pp_choice(dataset, 'feature', default_mode, i_round))
    return pipeline


def __get_pp_choice(dataset, category, default_mode, i_round):
    # select a solution bewteen the list of potential solutions from the category
    choices = __get_pp_list(dataset, category, default_mode)
    i_pp = i_round % len(choices)
    ref = choices[i_pp]
    s = pp_solutions_map[ref]
    if default_mode:
        params = s.default_params
    else:
        params = get_random_params(s.space_params)
    return ref, category, s.name, params


def __get_pp_list(dataset, category, default_mode):
    # generates the list of potential pre-processing choices in a specific category, depending on the problem type
    choices = []
    for s in pp_solutions:
        if s.pp_type == category and dataset.n_rows < s.limit_size:
            if default_mode and s.default_solution:
                choices.append(s.ref)
            elif not default_mode:
                choices.append(s.ref)
    return choices


def __get_model_class_list(dataset, level):
    # generates the list of potential models depending on the problem type
    choices = []
    for s in model_solutions:
        # check problem type
        if s.problem_type == '*' or s.problem_type == dataset.problem_type:
            # check level, limits and selectable
            if s.level == level and dataset.n_rows < s.limit_size:
                if s.selectable:
                    choices.append(s.ref)
    return choices


def __process_result(msg_result):
    dataset_id = msg_result['dataset_id']
    # update search history
    rpush_key_store('dataset:%s:rounds' % dataset_id, msg_result)

    # get search history
    df = get_search_rounds(dataset_id)

    dataset = get_dataset(dataset_id)
    search = __find_search_store(dataset)

    # get outlier threshold of metrics
    if len(df) > 10:
        search['threshold'] = __get_outlier_threshold(df)
    else:
        search['threshold'] = 0

    # check patience
    best, last_best = __get_last_best(df, search['level'])
    len_search = len(df[df.level == search['level']])
    if len_search - last_best > 100:
        if search['level'] == 1:
            # from level 1 move to level 2
            print('patience reached on level 1. searching now on ensembles')
            search['level'] = 2
            search['start'] = len(df)
            search['choices'] = __get_model_class_list(dataset, 2)
        else:
            # then we have finished: set status as completed
            print('patience reached on level 2. search completed')
            set_key_store('dataset:%s:status' % dataset_id, 'completed')
            return

    # restrict choice list
    if len_search - search['start'] > 50:
        if len(search['choices']) > 1:
            # reduce list by 2
            n = int(len(search['choices']) / 2)
            print('focusing list of models to %d models' % n)
            search['start'] = len_search
            search['choices'] = __get_list_best_models(df, search['level'])[:n]

    # generate graphs
    best = __get_best_models(df)
    best1 = best[best.level == 1]
    best2 = best[best.level == 2]
    graph_history_search(dataset, df, best1, 1)
    graph_history_search(dataset, df, best2, 2)

    # then update search
    set_key_store('dataset:%s:search' % dataset.dataset_id, search)
    set_key_store('dataset:%s:results' % dataset_id, len(df))
    set_key_store('dataset:%s:level' % dataset_id, search['level'])
    set_key_store('dataset:%s:best' % dataset_id, best.to_dict(orient='records'))
    set_key_store('dataset:%s:best_pp' % dataset_id, __get_best_pp(df))


def __get_outlier_threshold(df):
    # calculates the threshold for outliers from the score history
    # TODO: adapt threshold with number of [successful] rounds
    df0 = df[(df.cv_max != METRIC_NULL) & (df.level == 1)].groupby('model_name', as_index=False).min()
    scores = np.sort(df0.cv_max.values)
    outlier = scores[int(len(scores) / 2)]
    print('outlier threshold set at: %.5f ' % outlier)
    return outlier


def __get_last_best(df, level):
    # returns last best value and its index
    best = METRIC_NULL
    i_best = -1
    for i, score in enumerate(df[df.level == level].cv_max.values):
        if score < best:
            best = score
            i_best = i
    return best, i_best


def __get_list_best_models(df, level):
    # returns best models
    if len(df) < 1:
        return []
    best = df[df.level == level].sort_values(by=['solution', 'cv_max']).groupby('solution',as_index=False).first().sort_values(by='cv_max')

    return list(best.solution.values)


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
