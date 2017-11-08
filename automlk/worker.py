import datetime
import gc
import os
import socket
from copy import deepcopy
from .config import *
from .context import HyperContext, XySet
from .dataset import get_dataset
from .graphs import graph_histogram_regression, graph_histogram_classification, graph_predict_regression, graph_predict_classification
from .solutions import *
from .store import *
from .monitor import heart_beep, init_timer_worker, start_timer_worker, stop_timer_worker
from .solutions_pp import pp_solutions_map


def launch_worker():
    """
    periodically pool the receiver queue for a search job

    :return:
    """
    init_timer_worker()
    while True:
        # poll queue
        msg_search = brpop_key_store('controller:search_queue')
        heart_beep('worker', msg_search)
        if msg_search != None:
            print('reveived %s' % msg_search)
            msg_search = {**msg_search, **{'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                           'host_name': socket.gethostname()}}
            start_timer_worker(msg_search['time_limit'])
            job_search(msg_search)
            stop_timer_worker()


def job_search(msg_search):
    """
    execute the search on the scope defined in the messsage msg

    :param msg_search: message with parameters of the searcg
    :return:
    """
    # load dataset
    dataset = get_dataset(msg_search['dataset_id'])

    # load train/eval/test data
    ds_ini = pickle.load(open(get_dataset_folder(msg_search['dataset_id']) + '/data/eval_set.pkl', 'rb'))
    context = HyperContext(dataset.problem_type, dataset.x_cols, dataset.cat_cols, dataset.text_cols,
                           dataset.missing_cols)
    if msg_search['level'] == 1:
        # pre-processing on level 1 only
        t_start = time.time()
        context, ds = __pre_processing(context, msg_search['pipeline'], deepcopy(ds_ini))
        t_end = time.time()
        msg_search['duration_process'] = int(t_end - t_start)
    else:
        msg_search['duration_process'] = 0
        ds = ds_ini

    solution = model_solutions_map[msg_search['solution']]
    model = solution.model(dataset, context, msg_search['model_params'], msg_search['round_id'])
    msg_search['model_class'] = model.__class__.__name__
    if msg_search['level'] == 2:
        pool = __get_pool_models(dataset, msg_search['ensemble_depth'])
    else:
        pool = None
    __search(dataset, solution, model, msg_search, ds, pool)


def __pre_processing(context, pipeline, ds):
    # performs the different pre-processing steps
    context.pipeline = pipeline
    for ref, category, name, params in pipeline:
        solution = pp_solutions_map[ref]
        p_class = solution.process
        process = p_class(context, params)
        print('executing process', category, name, process.params)
        ds.X_train = process.fit_transform(ds.X_train, ds.y_train)
        ds.X_test = process.transform(ds.X_test)
        ds.X = process.transform(ds.X)
        if len(ds.X_submit) > 0:
            ds.X_submit = process.transform(ds.X_submit)
        print('-> %d features' % len(context.feature_names))

    return context, ds


def __search(dataset, solution, model, msg_search, ds, pool):
    print('optimizing with %s, params: %s' % (solution.name, model.params))
    # fit, test & score
    t_start = time.time()
    if msg_search['level'] == 2:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = model.cv_pool(pool, ds, msg_search['threshold'],
                                                                                        msg_search['ensemble_depth'])
    else:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = model.cv(ds, msg_search['threshold'])
    msg_search['num_rounds'] = model.num_rounds

    # check outlier
    if outlier:
        print('outlier, skipping this round')
        return

    # y_pred_eval as concat of folds
    y_pred_eval = np.concatenate(y_pred_eval_list)

    # reindex eval to be aligned with y
    y_pred_eval[ds.i_eval] = y_pred_eval.copy()

    # generate submit file
    if dataset.filename_submit != '':
        if dataset.problem_type == 'regression':
            submit = np.concatenate((ds.id_submit, y_pred_submit), axis=1)
        else:
            l = len(ds.id_submit)
            submit = np.concatenate((np.reshape(ds.id_submit, (l, 1)), np.reshape(y_pred_submit[:, 1], (l, 1))), axis=1)
        df_submit = pd.DataFrame(submit)
        df_submit.columns = [dataset.col_submit, dataset.y_col]
        # allocate id column to avoid type conversion (to float)
        df_submit[dataset.col_submit] = np.reshape(ds.id_submit, (l, 1))
        df_submit.to_csv(get_dataset_folder(dataset.dataset_id) + '/submit/submit_%s.csv' % model.round_id, index=False)

    # save model importance, prediction and model
    model.save_importance()
    model.save_predict(y_pred_eval, y_pred_test)
    # model.save_model()

    # generate graphs
    if dataset.problem_type == 'regression':
        graph_predict_regression(dataset, msg_search['round_id'], ds.y_train, y_pred_eval, 'eval')
        graph_predict_regression(dataset, msg_search['round_id'], ds.y_test, y_pred_test, 'test')
        graph_histogram_regression(dataset, msg_search['round_id'], y_pred_eval, 'eval')
        graph_histogram_regression(dataset, msg_search['round_id'], y_pred_test, 'test')
    else:
        graph_predict_classification(dataset, msg_search['round_id'], ds.y_train, y_pred_eval, 'eval')
        graph_predict_classification(dataset, msg_search['round_id'], ds.y_test, y_pred_test, 'test')
        graph_histogram_classification(dataset, msg_search['round_id'], y_pred_eval, 'eval')
        graph_histogram_classification(dataset, msg_search['round_id'], y_pred_test, 'test')


    t_end = time.time()
    msg_search['duration_model'] = int(t_end - t_start)
    __evaluate_round(dataset, msg_search, ds.y_train, y_pred_eval, ds.y_test, y_pred_test, ds.y_eval_list, y_pred_eval_list)


def __evaluate_round(dataset, msg_search, y_train, y_pred_eval, y_test, y_pred_test, y_eval_list, y_pred_eval_list):
    # score on full eval set, test set and cv
    msg_search['score_eval'] = dataset.evaluate_metric(y_train, y_pred_eval)
    msg_search['score_test'] = dataset.evaluate_metric(y_test, y_pred_test)
    msg_search['scores_cv'] = [dataset.evaluate_metric(y_act, y_pred) for y_act, y_pred in
                               zip(y_eval_list, y_pred_eval_list)]
    msg_search['cv_mean'] = np.mean(msg_search['scores_cv'])
    msg_search['cv_std'] = np.std(msg_search['scores_cv'])
    msg_search['cv_max'] = np.max(msg_search['scores_cv'])

    # score with secondary metrics
    msg_search['eval_other_metrics'] = {m: dataset.evaluate_metric(y_train, y_pred_eval, m) for m in
                                        dataset.other_metrics}
    msg_search['test_other_metrics'] = {m: dataset.evaluate_metric(y_test, y_pred_test, m) for m in
                                        dataset.other_metrics}

    rpush_key_store(RESULTS_QUEUE, msg_search)
    print('completed search:', msg_search)


def __get_pool_models(dataset, depth):
    # retrieves all results in order to build and ensemble
    df = get_search_rounds(dataset.dataset_id)

    # keep only the first (depth) models of level 0
    df = df[(df.level == 1) & (df.score_eval != METRIC_NULL)].sort_values(by=['model_name', 'score_eval'])
    round_ids = []
    model_names = []
    k_model = ''
    for index, row in df.iterrows():
        if k_model != row['model_name']:
            count_model = 0
            k_model = row['model_name']
        if count_model > depth:
            continue
        model_names.append(row['model_name'])
        round_ids.append(row['round_id'])
        count_model += 1

    print('length of pool: %d for ensemble of depth %d' % (len(round_ids), depth))
    # retrieves predictions
    preds = [get_pred_eval_test(dataset.dataset_id, round_id) for round_id in round_ids]
    preds_eval = [x[0] for x in preds]
    preds_test = [x[1] for x in preds]

    # TODO replace submit
    return EnsemblePool(round_ids, model_names, preds_eval, preds_test, preds_test)


def __store_search_error(dataset, t, e, model):
    print('Error: ', e)
    # track error
    with open(get_dataset_folder(dataset.dataset_id) + '/errors.txt', 'a') as f:
        f.write("'time':'%s', 'model': %s, 'params': %s, '\n Error': %s \n" % (
            t, model.model_name, model.params, str(e)))


def get_search_rounds(dataset_id):
    """
    get all the results of the search with preprocessing and models

    :param dataset_id: id of the dataset
    :return: results of the search as a dataframe
    """
    results = list_key_store('dataset:%s:rounds' % dataset_id)
    return pd.DataFrame(results)
