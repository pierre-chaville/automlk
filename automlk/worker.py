import eli5
import threading
import _thread
import sys
import os
from copy import deepcopy
from .config import *
from .dataset import get_dataset, get_dataset_status
from .graphs import graph_histogram_regression, graph_histogram_classification, graph_predict_regression, \
    graph_predict_classification
from .prepare import get_eval_sets
from .solutions import *
from .monitor import *
from .metrics import evaluate_metric
from .specific import apply_specific_metrics, return_specific_metrics
from .solutions_pp import pp_solutions_map
from .results import get_pred_eval_test
from .xyset import XySet
from sklearn.pipeline import make_pipeline


log = logging.getLogger(__name__)


def get_search_rounds(dataset_id):
    """
    get all the results of the search with preprocessing and models

    :param dataset_id: id of the dataset
    :return: results of the search as a dataframe
    """
    results = list_key_store('dataset:%s:rounds' % dataset_id)
    return pd.DataFrame(results)


def __timer_control(f_stop):
    global __worker_timer_start
    global __worker_timer_limit
    global __worker_dataset

    t = time.time()
    # check if duration > max
    if (__worker_timer_limit > 0) and (t - __worker_timer_start > __worker_timer_limit):
        f_stop.set()
        log.info('max delay %d seconds reached...' % __worker_timer_limit)
        _thread.interrupt_main()

    # check dataset is in searching model
    if __worker_dataset != '':
        if get_dataset_status(__worker_dataset) != 'searching':
            f_stop.set()
            log.info('dataset %s is no more in searching mode, aborting...' % __worker_dataset)
            _thread.interrupt_main()

    if not f_stop.is_set():
        # call again in 10 seconds
        threading.Timer(10, __timer_control, [f_stop]).start()


def worker_loop(worker_id, gpu=False):
    """
    periodically pool the receiver queue for a search job
    :param worker_id: index of the worker on this machine
    :param gpu: can use gpu on this machine
    :return:
    """
    global __worker_timer_start
    global __worker_timer_limit
    global __worker_dataset
    __worker_dataset = ''
    __worker_timer_start = 0
    __worker_timer_limit = 0
    f_stop = threading.Event()
    # start calling f now and every 60 sec thereafter
    __timer_control(f_stop)
    while True:
        try:
            # poll queue
            msg_search = brpop_key_store('controller:search_queue')
            heart_beep('worker', msg_search, worker_id, gpu)
            __worker_timer_start = time.time()
            __worker_timer_limit = 0
            __worker_dataset = ''
            if msg_search is not None:
                __worker_dataset = msg_search['dataset_id']
                __worker_timer_limit = msg_search['time_limit']
                log.info('received %s' % msg_search)
                msg_search = {**msg_search, **{'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                               'host_name': socket.gethostname()}}
                job_search(msg_search)
        except KeyboardInterrupt:
            log.info('Keyboard interrupt: exiting')
            # stop the timer thread
            f_stop.set()
            exit()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            log.error('%s in %s line:%s error: %s' % (exc_type.__name__, fname, str(exc_tb.tb_lineno), str(e)))
            with open(get_data_folder() + '/errors.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + str(msg_search) + '\n')
                f.write('%s in %s line:%s error: %s' % (exc_type.__name__, fname, str(exc_tb.tb_lineno), str(e)) + '\n')
                f.write('-'*80 + '\n')


def job_search(msg_search):
    """
    execute the search on the scope defined in the messsage msg

    :param msg_search: message with parameters of the searcg
    :return:
    """
    # load dataset
    dataset = get_dataset(msg_search['dataset_id'])

    # define function for specific metric if any
    if dataset.metric == 'specific':
        apply_specific_metrics(dataset.dataset_id)

    # load train/eval/test data
    ds_ini = get_eval_sets(msg_search['dataset_id'])

    if msg_search['level'] == 2:
        ds_ini = __create_stacking(dataset, __get_pool_models(dataset, msg_search['ensemble_depth']), ds_ini)

    # pre-processing
    t_start = time.time()
    feature_names, ds, pipe = __pre_processing(dataset, msg_search['pipeline'], deepcopy(ds_ini))
    t_end = time.time()
    msg_search['duration_process'] = int(t_end - t_start)

    # generate model from solution
    solution = model_solutions_map[msg_search['solution']]
    if solution.is_wrapper:
        model = solution.model(**{**{'problem_type': dataset.problem_type,
                                     'y_n_classes': dataset.y_n_classes},
                                  **msg_search['model_params']})
    else:
        model = solution.model(**msg_search['model_params'])
    msg_search['model_class'] = model.__class__.__name__
    pipe_transform, pipe_model = make_pipeline(*pipe), make_pipeline(*(pipe + [model]))

    # then proceed to the search
    __search(dataset, feature_names, solution, pipe_transform, pipe_model, model, msg_search, ds)


def __pre_processing(dataset, pipeline, ds):
    # performs the different pre-processing steps
    pipe = []
    feature_names = None
    p_context = [{'name': f.name, 'col_type': f.col_type, 'raw_type': f.raw_type, 'n_missing': int(f.n_missing),
                  'n_unique_values': int(f.n_unique_values), 'text_ref': f.text_ref}
                 for f in dataset.features if f.name in dataset.x_cols]
    for ref, category, name, params in pipeline:
        if category != 'sampling':
            solution = pp_solutions_map[ref]
            p_class = solution.process
            process = p_class(**{**params, 'context': p_context})
            log.info('executing process %s %s %s' % (category, name, process.transformer_params))
            ds.X_train = process.fit_transform(ds.X_train, ds.y_train)
            pipe.append(process)
            ds.X_test = process.transform(ds.X_test)
            ds.X = process.transform(ds.X)
            if len(ds.X_submit) > 0:
                ds.X_submit = process.transform(ds.X_submit)
            log.info('-> %d features (%s)' % (len(process.get_feature_names()), type(ds.X_train)))
            feature_names = process.get_feature_names()
    return feature_names, ds, pipe


def __search(dataset, feature_names, solution, pipe_transform, pipe_model, model, msg_search, ds):
    log.info('optimizing with %s, params: %s' % (solution.name, msg_search['model_params']))
    # fit, test & score
    t_start = time.time()
    round_id = msg_search['round_id']
    level = msg_search['level']
    threshold = msg_search['threshold']
    pct = msg_search['pct']
    cv = msg_search['cv']

    outlier, y_pred_eval_list, y_pred_test, y_pred_submit, ds = __cross_validation(solution, model, dataset, ds, threshold, pct, cv)

    if hasattr(model, 'num_rounds'):
        msg_search['num_rounds'] = model.num_rounds
    else:
        msg_search['num_rounds'] = None

    # check outlier
    if outlier:
        log.info('outlier, skipping this round')
        return

    # save model importance
    if level == 2 and solution.is_wrapper:
        __save_importance(model.model, dataset, feature_names, round_id)
    else:
        __save_importance(model, dataset, feature_names, round_id)

    # create eval and submit results
    y_pred_eval = __create_eval_submit(dataset, ds, round_id, y_pred_eval_list, y_pred_submit, cv)

    # save predictions (eval and test set)
    pickle.dump([y_pred_eval, y_pred_test, y_pred_submit],
                open(get_dataset_folder(dataset.dataset_id) + '/predict/%s.pkl' % round_id, 'wb'))

    # then feature names
    pickle.dump(ds.X_train.columns, open(
        get_dataset_folder(dataset.dataset_id) + '/models/' + '%s_feature_names.pkl' % msg_search['round_id'], 'wb'))

    # generate graphs
    __create_graphs(dataset, round_id, ds, y_pred_eval, y_pred_test)

    t_end = time.time()
    msg_search['duration_model'] = int(t_end - t_start)

    # calculate metrics
    __evaluate_round(dataset, msg_search, ds.y_train, y_pred_eval, ds.y_test, y_pred_test, ds.y_eval_list,
                     y_pred_eval_list)

    # then save model, pipe
    __save_model(dataset, round_id, pipe_transform, pipe_model, model)

    # explain model
    __explain_model(dataset, msg_search['round_id'], pipe_model, model, feature_names)


def __cross_validation(solution, model, dataset, ds, threshold, pct, cv):
    # performs a cross validation on cv_folds, and predict also on X_test
    y_pred_eval, y_pred_test, y_pred_submit = [], [], []
    for i, (train_index, eval_index) in enumerate(ds.cv_folds):
        # use only a percentage of data (default is 100% )
        train_index1 = train_index[:int(len(train_index)*pct)]
        X1, y1 = ds.X_train.iloc[train_index1], ds.y_train[train_index1]
        X2, y2 = ds.X_train.iloc[eval_index], ds.y_train[eval_index]
        if i == 0 and solution.use_early_stopping:
            log.info('early stopping round')
            if __fit_early_stopping(solution, model, dataset, threshold, X1, y1, X2, y2):
                return True, 0, 0, 0, ds

        # then train on train set and predict on eval set
        model.fit(X1, y1)
        y_pred = __predict(solution, model, X2)

        if threshold != 0:
            # test outlier:
            score = __evaluate_metric(dataset, y2, y_pred)
            if score > threshold:
                log.info('%dth round found outlier: %.5f with threshold %.5f' % (i, score, threshold))
                return True, 0, 0, 0, ds

        y_pred_eval.append(y_pred)

        # we also predict on test & submit set (to be averaged later)
        y_pred_test.append(__predict(solution, model, ds.X_test))

        if not cv:
            # we stop at the first fold
            y_pred_test = y_pred_test[0]
            if dataset.mode == 'competition':
                y_pred_submit = __predict(solution, model, ds.X_submit)
            # update y_train on fold, in order to compute metrics and graphs
            ds.y_train = y2
            return False, y_pred_eval, y_pred_test, y_pred_submit, ds

    if dataset.mode == 'standard':
        # train on complete train set
        model.fit(ds.X_train, ds.y_train)
        y_pred_test = __predict(solution, model, ds.X_test)
    else:
        # train on complete X y set
        model.fit(ds.X, ds.y)
        if dataset.mode == 'competition':
            y_pred_submit = __predict(solution, model, ds.X_submit)
            # test = mean of y_pred_test on multiple folds
            y_pred_test = np.mean(y_pred_test, axis=0)
        else:
            y_pred_test = __predict(solution, model, ds.X_test)

    return False, y_pred_eval, y_pred_test, y_pred_submit, ds


def __create_stacking(dataset, pool, ds):
    # create X by stacking predictions
    for j, (u, m, p_eval, p_test, p_submit) in enumerate(
            zip(pool.pool_model_round_ids, pool.pool_model_names, pool.pool_eval_preds,
                pool.pool_test_preds, pool.pool_submit_preds)):
        # check if array has 2 dimensions
        shape = len(np.shape(p_eval))
        if shape == 1:
            p_eval = np.reshape(p_eval, (len(p_eval), 1))
            p_test = np.reshape(p_test, (len(p_test), 1))
            if dataset.mode == 'competition':
                p_submit = np.reshape(p_submit, (len(p_submit), 1))
        if j == 0:
            ds.X_train = p_eval
            ds.X_test = p_test
            if dataset.mode == 'competition':
                ds.X_submit = p_submit
        else:
            # stack vertically the predictions
            ds.X_train = np.concatenate((ds.X_train, p_eval), axis=1)
            ds.X_test = np.concatenate((ds.X_test, p_test), axis=1)
            if dataset.mode == 'competition':
                ds.X_submit = np.concatenate((ds.X_submit, p_submit), axis=1)

    # then convert to dataframes
    ds.X_train, ds.X_test, ds.X_submit = pd.DataFrame(ds.X_train), pd.DataFrame(ds.X_test), pd.DataFrame(ds.X_submit)

    # update feature names
    feature_names = __get_pool_features(dataset, pool)
    ds.X_train.columns = feature_names
    if len(ds.X_test) > 0:
        ds.X_test.columns = feature_names
    if len(ds.X_submit) > 0:
        ds.X_submit.columns = feature_names

    # X and y for fit
    ds.X = pd.concat([ds.X_train, ds.X_test])
    ds.y = np.concatenate((ds.y_train, ds.y_test))

    return ds


def __create_eval_submit(dataset, ds, round_id, y_pred_eval_list, y_pred_submit, cv):
    # create eval and submit results from lists

    if cv:
        # y_pred_eval as concat of folds
        y_pred_eval = np.concatenate(y_pred_eval_list)
        # reindex eval to be aligned with y
        y_pred_eval[ds.i_eval] = y_pred_eval.copy()
    else:
        y_pred_eval = y_pred_eval_list[0]

    # generate submit file
    if dataset.filename_submit != '':
        ls = len(ds.id_submit)
        # if dataset.problem_type == 'regression':
        if np.shape(y_pred_submit)[1] == 1:
            submit = np.concatenate((np.reshape(ds.id_submit, (ls, 1)), np.reshape(y_pred_submit, (ls, 1))), axis=1)
        else:
            submit = np.concatenate((np.reshape(ds.id_submit, (ls, 1)), np.reshape(y_pred_submit[:, 1], (ls, 1))),
                                    axis=1)
        # create submission file

        df_submit = pd.DataFrame(submit)
        df_submit.columns = [dataset.col_submit, dataset.y_col]
        # allocate id column to avoid type conversion (to float)
        df_submit[dataset.col_submit] = np.reshape(ds.id_submit, (ls, 1))
        df_submit.to_csv(get_dataset_folder(dataset.dataset_id) + '/submit/submit_%s.csv' % round_id, index=False)

    return y_pred_eval


def __create_graphs(dataset, round_id, ds, y_pred_eval, y_pred_test):
    # generate graphs
    if dataset.problem_type == 'regression':
        graph_predict_regression(dataset, round_id, ds.y_train, y_pred_eval, 'eval')
        graph_predict_regression(dataset, round_id, ds.y_test, y_pred_test, 'test')
        graph_histogram_regression(dataset, round_id, y_pred_eval, 'eval')
        graph_histogram_regression(dataset, round_id, y_pred_test, 'test')
    else:
        graph_predict_classification(dataset, round_id, ds.y_train, y_pred_eval, 'eval')
        graph_predict_classification(dataset, round_id, ds.y_test, y_pred_test, 'test')
        graph_histogram_classification(dataset, round_id, y_pred_eval, 'eval')
        graph_histogram_classification(dataset, round_id, y_pred_test, 'test')


def __fit_early_stopping(solution, model, dataset, threshold, X1, y1, X2, y2):
    # fit with early stopping the model
    if solution.is_wrapper:
        # with early stopping, we perform an initial round to get number of rounds
        model.fit_early_stopping(X1, y1, X2, y2)
    else:
        model.fit(X1, y1, eval_set=[(X2, y2)], early_stopping_rounds=PATIENCE, verbose=False)
        if solution.early_stopping == 'LGBM':
            num_rounds = model.best_iteration_ if model.best_iteration_ != 0 else MAX_ROUNDS
        elif solution.early_stopping == 'XGB':
            num_rounds = model.best_iteration if model.best_iteration != 0 else MAX_ROUNDS
        params = model.get_params()
        params['n_estimators'] = num_rounds
        model.set_params(**params)
        log.info('early stopping best iteration = %d' % num_rounds)

    if threshold != 0:
        # test outlier (i.e. exceeds threshold)
        y_pred = __predict(solution, model, X2)
        score = __evaluate_metric(dataset, y2, y_pred)
        if score > threshold:
            log.info('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
            time.sleep(10)
            return True
    return False


def __predict(solution, model, X):
    if solution.problem_type == 'regression':
        return model.predict(X)
    else:
        return model.predict_proba(X)


def __resample(pipeline, X, y):
    # apply resampling steps in pipeline
    for ref, category, name, params in pipeline:
        if category == 'sampling':
            solution = pp_solutions_map[ref]
            p_class = solution.process
            process = p_class(params)
            return process.fit_sample(X, y)
    return X, y


def __save_importance(model, dataset, feature_names, round_id):
    # saves feature importance (as a dataframe)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame(feature_names)
        importance['importance'] = model.feature_importances_
        importance.columns = ['feature', 'importance']
        pickle.dump(importance, open(get_dataset_folder(dataset.dataset_id) + '/features/%s.pkl' % round_id, 'wb'))
    elif hasattr(model, 'dict_importance_'):
        # xgboost type: feature importance is a dictionary
        imp = model.dict_importance_
        importance = pd.DataFrame([{'feature': key, 'importance': imp[key]} for key in imp.keys()])
        pickle.dump(importance, open(get_dataset_folder(dataset.dataset_id) + '/features/%s.pkl' % round_id, 'wb'))


def __evaluate_round(dataset, msg_search, y_train, y_pred_eval, y_test, y_pred_test, y_eval_list, y_pred_eval_list):
    # score on full eval set, test set and cv
    msg_search['score_eval'] = __evaluate_metric(dataset, y_train, y_pred_eval)
    msg_search['score_test'] = __evaluate_metric(dataset, y_test, y_pred_test)
    msg_search['scores_cv'] = [__evaluate_metric(dataset, y_act, y_pred) for y_act, y_pred in
                               zip(y_eval_list, y_pred_eval_list)]
    msg_search['cv_mean'] = np.mean(msg_search['scores_cv'])
    msg_search['cv_std'] = np.std(msg_search['scores_cv'])
    msg_search['cv_max'] = np.max(msg_search['scores_cv'])

    # score with secondary metrics
    msg_search['eval_other_metrics'] = {m: __evaluate_other_metrics(dataset, m, y_train, y_pred_eval) for m in
                                        dataset.other_metrics}
    msg_search['test_other_metrics'] = {m: __evaluate_other_metrics(dataset, m, y_test, y_pred_test) for m in
                                        dataset.other_metrics}

    rpush_key_store(RESULTS_QUEUE, msg_search)
    log.info('completed search')


def __get_pool_features(dataset, pool):
    # return the lst of features in an ensemble model
    if dataset.problem_type == 'regression':
        feature_names = [name + '_' + str(round_id) for round_id, name in
                         zip(pool.pool_model_round_ids, pool.pool_model_names)]
    else:
        feature_names = []
        for round_id, name in zip(pool.pool_model_round_ids, pool.pool_model_names):
            for k in range(dataset.y_n_classes):
                feature_names.append(name + '_' + str(k) + '_' + str(round_id))
    return feature_names


def __get_pool_models(dataset, depth):
    # retrieves all results in order to build and ensemble
    df = get_search_rounds(dataset.dataset_id)

    # keep only the first (depth) models of level 0
    df = df[((df.level == 1) & (df.score_eval != METRIC_NULL)) & df.cv].sort_values(by=['model_name', 'score_eval'])
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

    log.info('length of pool: %d for ensemble of depth %d' % (len(round_ids), depth))

    # retrieves predictions
    preds = [get_pred_eval_test(dataset.dataset_id, round_id) for round_id in round_ids]

    # exclude predictions with nan
    excluded = [i for i, x in enumerate(preds) if not (np.max(x[0]) == np.max(x[0]))]

    preds = [x for i, x in enumerate(preds) if i not in excluded]
    round_ids = [x for i, x in enumerate(round_ids) if i not in excluded]
    model_names = [x for i, x in enumerate(model_names) if i not in excluded]

    preds_eval = [x[0] for x in preds]
    preds_test = [x[1] for x in preds]
    preds_submit = [x[2] for x in preds]

    return EnsemblePool(round_ids, model_names, preds_eval, preds_test, preds_submit)


def __store_search_error(dataset, t, e, model):
    log.info('Error: %s' % e)
    # track error


def __evaluate_metric(dataset, y_act, y_pred):
    """
    evaluates primary metrics for the dataset

    :param dataset: dataset object
    :param y_act: actual values
    :param y_pred: predicted values
    :return: metrics
    """
    if dataset.metric == 'specific':
        if dataset.best_is_min:
            return return_specific_metrics(y_act, y_pred)
        else:
            return -return_specific_metrics(y_act, y_pred)
    else:
        return evaluate_metric(y_act, y_pred, dataset.metric, dataset.y_n_classes)


def __evaluate_other_metrics(dataset, m, y_act, y_pred):
    """
    evaluates other metrics for the dataset

    :param dataset: dataset object
    :param m: name of the other metric
    :param y_act: actual values
    :param y_pred: predicted values
    :return: metrics
    """
    return evaluate_metric(y_act, y_pred, m, dataset.y_n_classes)


def __save_model(dataset, round_id, pipe_transform, pipe_model, model):
    """
    save model, pipe

    :param dataset: dataset object
    :param round_id: round id
    :param pipe_transform: sklearn pipeline of pre-processing steps only
    :param pipe_model: sklearn pipeline of pre-processing steps + model
    :param model: estimator model
    :return:
    """
    folder = get_dataset_folder(dataset.dataset_id) + '/models/'
    pickle.dump(model, open(folder + '%s_model.pkl' % round_id, 'wb'))
    pickle.dump(pipe_model, open(folder + '%s_pipe_model.pkl' % round_id, 'wb'))
    pickle.dump(pipe_transform, open(folder + '%s_pipe_transform.pkl' % round_id, 'wb'))


def __explain_model(dataset, round_id, pipe_model, model, feature_names):
    """
    explain the weights and the prediction of the model

    :param dataset: dataset
    :param round_id: round if
    :param pipe_model: the pipeline including the model
    :param model: the model only
    :param feature_names: feature names
    :return:
    """
    try:
        exp = eli5.explain_weights(model, feature_names=list(feature_names))
        with open(get_dataset_folder(dataset.dataset_id) + '/predict/eli5_model_%s.html' % round_id, 'w') as f:
            f.write(eli5.format_as_html(exp))
    except:
        return