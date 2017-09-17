import datetime
import gc
import os
import socket
from .preprocessing import pre_processing
from .context import HyperContext
from .dataset import get_dataset
from .graphs import graph_pred_histogram, graph_predict
from .solutions import *
from .store import *

PATIENCE_RANDOM = 100
PATIENCE_ENSEMBLE = 100


def worker_search(dataset_id, search_mode='auto', max_iter=500):
    """
    launch the search of the best preprocessing + models

    :param dataset_id: id of the dataset
    :param search_mode: options are : 'default' = default parameters, 'random' = random search, 'auto' = default, then random
    'ensemble' = ensemble models (requires a minimum of models to be searched in random/auto mode before)
    :param max_iter: maximum number of searches
    """
    # performs a random search on all models

    # load dataset
    dataset = get_dataset(dataset_id)

    # load train/eval/test data
    X_train_ini, X_test_ini, y_train_ini, y_test_ini, cv_folds, y_eval_list, y_eval, i_eval = pickle.load(
        open(get_dataset_folder(dataset_id) + '/data/eval_set.pkl', 'rb'))

    print('hyper optimizing...')
    for i in range(max_iter):

        # get search history
        df = get_search_rounds(dataset.dataset_id)

        # get outlier threshold of metrics
        if len(df) > 10:
            threshold = __get_outlier_threshold(df)
        else:
            threshold = 0

        store = {'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 'host_name': socket.gethostname(), 'dataset_id': dataset.dataset_id, 'search_mode': search_mode,
                 'round_id': str(incr_key_store('dataset:' + dataset_id + ':round_counter'))}

        t_start = time.time()
        context = HyperContext(dataset.problem_type, dataset.x_cols, dataset.cat_cols, dataset.missing_cols)

        context, X_train, y_train, X_test, y_test = pre_processing(context, X_train_ini, y_train_ini, X_test_ini,
                                                                   y_test_ini)

        process_steps = {p.process_name: p.params for p in context.pipeline}
        t_end = time.time()
        store['duration_process'] = int(t_end - t_start)
        print('preprocessing steps:', process_steps)

        # TODO: mode focus = overweight choice of best models after some rounds
        # try:
        level = 1
        pool = None
        if search_mode == 'default':
            solution, model = __get_default_model(dataset, context, store['round_id'])
            if not model:
                return
        elif search_mode == 'random':
            solution, model = __get_random_model(dataset, context, store['round_id'], 1)
        elif search_mode == 'ensemble':
            ensemble_depth = random.randint(1, 10)
            pool = __get_pool_models(dataset, ensemble_depth)
            solution, model = __get_random_model(dataset, context, store['round_id'], 2)
            level = 2
        elif search_mode == 'auto':
            # search in default mode, then random
            solution, model = __get_default_model(dataset, context, store['round_id'])
            if not model:
                best, i_best = __get_last_best(df, level=0)
                if len(df) - i_best > PATIENCE_RANDOM:
                    # ensemble search
                    level = 2
                    search_mode = 'ensemble'
                    best, i_best = __get_last_best(df, level=1)
                    if i_best > 0 and len(df) - i_best > PATIENCE_ENSEMBLE:
                        print('ensemble search complete - patience reached')
                        return
                    ensemble_depth = random.randint(1, 10)
                    pool = __get_pool_models(dataset, ensemble_depth)
                    solution, model = __get_random_model(dataset, context, store['round_id'], 2)
                    store['ensemble_depth'] = ensemble_depth
                else:
                    # random search
                    solution, model = __get_random_model(dataset, context, store['round_id'], 1)

        store['level'] = level
        store['threshold'] = threshold
        store['model_name'] = solution.name
        store['model_ref'] = solution.ref
        store['model_class'] = model.__class__.__name__
        store['process_steps'] = process_steps
        store['model_params'] = model.params

        __search(dataset, solution, model, store, X_train, y_train, X_test, y_test, y_eval_list, i_eval, cv_folds, pool)


def __search(dataset, solution, model, store, X_train, y_train, X_test, y_test, y_eval_list, i_eval, cv_folds, pool):

    print('optimizing with %s, params: %s' % (solution.name, model.params))

    # fit, test & score
    t_start = time.time()
    if store['level'] == 2:
        outlier, y_pred_eval_list, y_pred_test_list = model.cv_pool(pool, y_train, y_test, cv_folds, store['threshold'],
        store['ensemble_depth'])
    else:
        outlier, y_pred_eval_list, y_pred_test_list = model.cv(X_train, y_train, X_test, y_test, cv_folds,
                                                               store['threshold'])
    store['num_rounds'] = model.num_rounds

    # check outlier
    if outlier:
        print('outlier, skipping this round')
        return

    # y_pred_eval as concat of folds
    y_pred_eval = np.concatenate(y_pred_eval_list)

    # reindex eval to be aligned with y
    y_pred_eval[i_eval] = y_pred_eval.copy()

    # mean of y_pred_test on multiple folds
    y_pred_test = np.mean(y_pred_test_list, axis=0)

    # save model importance, prediction and model
    model.save_importance()
    model.save_predict(y_pred_eval, y_pred_test)
    # model.save_model()

    # generate graphs
    graph_predict(dataset, store['round_id'], y_train, y_pred_eval, 'eval')
    graph_predict(dataset, store['round_id'], y_test, y_pred_test, 'test')
    graph_pred_histogram(dataset.dataset_id, store['round_id'], y_pred_eval, 'eval')
    graph_pred_histogram(dataset.dataset_id, store['round_id'], y_pred_test, 'test')

    t_end = time.time()
    store['duration_model'] = int(t_end - t_start)
    __evaluate_round(dataset, store, y_train, y_pred_eval, y_test, y_pred_test, y_eval_list, y_pred_eval_list)


def __evaluate_round(dataset, store, y_train, y_pred_eval, y_test, y_pred_test, y_eval_list, y_pred_eval_list):

    # score on full eval set, test set and cv
    store['score_eval'] = dataset.evaluate_metric(y_train, y_pred_eval)
    store['score_test'] = dataset.evaluate_metric(y_test, y_pred_test)
    store['scores_cv'] = [dataset.evaluate_metric(y_act, y_pred) for y_act, y_pred in zip(y_eval_list, y_pred_eval_list)]
    store['cv_mean'] = np.mean(store['scores_cv'])
    store['cv_std'] = np.std(store['scores_cv'])

    # score with secondary metrics
    store['eval_other_metrics'] = {m: dataset.evaluate_metric(y_train, y_pred_eval, m) for m in dataset.other_metrics}
    store['test_other_metrics'] = {m: dataset.evaluate_metric(y_test, y_pred_test, m) for m in dataset.other_metrics}

    store_search_round(dataset.dataset_id, store)


def __get_outlier_threshold(df):
    # calculates the threshold for outliers from the score history
    # TODO: adapt threshold with number of [successful] rounds
    df0 = df[(df.score_eval != METRIC_NULL) & (df.model_level == 1)].groupby('model', as_index=False).min()
    score_mean = df0.score_eval.mean()
    score_std = df0.score_eval.std()
    print('outlier threshold set at: %.5f (mean: %.5f, std: %.5f)' % (
        abs(score_mean + 0.5 * score_std), abs(score_mean), abs(score_std)))
    return score_mean + 3 * score_std


def __get_last_best(df, level):
    # returns last best value and its index
    best = METRIC_NULL
    i_best = -1
    for i, score in enumerate(df[df.model_level == level].score_eval.values):
        if score < best:
            best = score
            i_best = i
    return best, i_best


def __get_model_class_list(dataset, level):
    # generates the list of potential models depending on the problem type
    choices = []
    for s in model_solutions:
        # check problem type
        if s.problem_type == '*' or s.problem_type == dataset.problem_type:
            # check level, limits and selectable
            if s.level == level and dataset.n_rows < s.limit_size:
                if s.selectable:
                    choices.append(s)
    return choices


def __get_default_model(dataset, context, round_id):
    # generates a default model to evaluate which has not been tested yet

    # retrieves the list of default models already searched
    default_file_name = get_dataset_folder(dataset.dataset_id) + '/default.txt'
    if os.path.isfile(default_file_name):
        with open(default_file_name, 'r') as f:
            searched_list = [line.rstrip() for line in f]
    else:
        searched_list = []

    # find a model not already in in the list
    choices = __get_model_class_list(dataset, 1)

    remaining_choices = [s for s in choices if s.ref not in searched_list]

    if len(remaining_choices) == 0:
        return None, None

    # take the first model in the remaining list
    solution = remaining_choices[0]
    model = solution.model(dataset, context, solution.default_params, round_id)

    # mark the file as reserved
    with open(default_file_name, 'a') as f:
        f.write(solution.ref + '\n')

    return solution, model


def __get_random_model(dataset, context, round_id, level):
    # generates a random model with random parameters
    choices = __get_model_class_list(dataset, level)
    solution = random.choice(choices)
    model = solution.model(dataset, context, get_random_params(solution.space_params), round_id)
    return solution, model


def __get_pool_models(dataset, round_id, depth):
    # retrieves all results in order to build and ensemble
    df = get_search_rounds(dataset.dataset_id)

    # keep only the first (depth) models of level 0
    df = df[(df.model_level == 0) & (df.score_eval != METRIC_NULL)].sort_values(by=['model', 'score_eval'])
    round_ids = []
    model_names = []
    k_model = ''
    for index, row in df.iterrows():
        if k_model != row['model']:
            count_model = 0
            k_model = row['model']
        if count_model > depth:
            continue
        model_names.append(row['model'])
        round_ids.append(row['round_id'])
        count_model += 1

    print('length of pool: %d for ensemble of depth %d' % (len(round_ids), depth))
    # retrieves predictions
    preds = [get_pred_eval_test(dataset.dataset_id, round_id) for round_id in round_ids]
    preds_eval = [x[0] for x in preds]
    preds_test = [x[1] for x in preds]

    return EnsemblePool(round_ids, model_names, preds_eval, preds_test)


def __store_search_error(dataset, t, e, model):
    print('Error: ', e)
    # track error
    with open(get_dataset_folder(dataset.dataset_id) + '/errors.txt', 'a') as f:
        f.write("'time':'%s', 'model': %s, 'params': %s, '\n Error': %s \n" % (
            t, model.model_name, model.params, str(e)))


def store_search_round(dataset_id, store):
    # track score
    rpush_key_store('dataset:%s:rounds' % dataset_id, store)
    """
    with open(get_dataset_folder(store['dataset_id']) + '/search.txt', 'a') as f:
        s = "{'time':'%s', 'duration_process':'%.2f', 'duration':'%.2f', 'score_eval':%.6f, 'score_test':%.6f, 'scores_cv': %s, 'cv_mean':%.6f, 'cv_std':%.6f, 'eval_other_metrics': %s, 'test_other_metrics': %s, 'model_class': '%s', 'model_level': %d, 'model_ref': '%s', 'model': '%s', 'params': %s, 'rounds': %d, 'process': %s, 'round_id': '%s', 'host': '%s', 'search_mode': '%s'}" % (
            store['start_time'], store['duration_process'], store['duration_model'], store['score_eval'],
            store['score_test'], store['scores_cv'], store['cv_mean'], store['cv_std'], store['eval_other_metrics'],
            store['test_other_metrics'], store['model_name'], store['level'], store['solution_ref'],
            store['solution_name'], store['model_params'], store['num_rounds'], store['process_steps'],
            store['round_id'], store['host_name'], store['search_mode'])

        f.write(s + '\n')
        print(s)
    """

def get_search_rounds(dataset_id):
    """
    get all the results of the search with preprocessing and models

    :param dataset_id: id of the dataset
    :return: results of the search as a dataframe
    """
    # return search logs as a dataframe
    """
    with open(get_dataset_folder(dataset_id) + '/search.txt', 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        results.append(eval(line))
    """
    results = list_key_store('dataset:%s:rounds' % dataset_id)
    return pd.DataFrame(results)
