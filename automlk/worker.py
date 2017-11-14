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
        try:
            # poll queue
            msg_search = brpop_key_store('controller:search_queue')
            heart_beep('worker', msg_search)
            if msg_search != None:
                print('received %s' % msg_search)
                msg_search = {**msg_search, **{'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                               'host_name': socket.gethostname()}}
                start_timer_worker(msg_search['time_limit'])
                job_search(msg_search)
                stop_timer_worker()
        except KeyboardInterrupt:
            print('Keyboard interrupt: exiting')
            exit()
        except Exception as e:
            print("Error", e)
            exit()


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
        context, ds, final_pipeline = __pre_processing(context, msg_search['pipeline'], deepcopy(ds_ini))
        t_end = time.time()
        msg_search['duration_process'] = int(t_end - t_start)
    else:
        msg_search['duration_process'] = 0
        ds = ds_ini
        final_pipeline = []

    solution = model_solutions_map[msg_search['solution']]
    model = solution.model(dataset, context, msg_search['model_params'], msg_search['round_id'])
    msg_search['model_class'] = model.__class__.__name__
    if msg_search['level'] == 2:
        pool = __get_pool_models(dataset, msg_search['ensemble_depth'])
    else:
        pool = None
    __search(dataset, solution, model, msg_search, ds, pool, final_pipeline)


def __pre_processing(context, pipeline, ds):
    # performs the different pre-processing steps
    context.pipeline = pipeline
    for ref, category, name, params in pipeline:
        if category != 'sampling':
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
    final_pipeline = [p for p in pipeline if p[1] == 'sampling']
    print('final pipeline', final_pipeline)
    return context, ds, final_pipeline


def __search(dataset, solution, model, msg_search, ds, pool, pipeline):
    print('optimizing with %s, params: %s' % (solution.name, model.params))
    # fit, test & score
    t_start = time.time()
    if msg_search['level'] == 2:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = model.cv_pool(pool, ds, msg_search['threshold'],
                                                                                        msg_search['ensemble_depth'])
    else:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = __cv(model, dataset, ds, pipeline, msg_search['threshold'])
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
            print('len id', l, 'y_pred_submit', np.shape(y_pred_submit))
            submit = np.concatenate((np.reshape(ds.id_submit, (l, 1)), np.reshape(y_pred_submit[:, 1], (l, 1))), axis=1)
        df_submit = pd.DataFrame(submit)
        df_submit.columns = [dataset.col_submit, dataset.y_col]
        # allocate id column to avoid type conversion (to float)
        df_submit[dataset.col_submit] = np.reshape(ds.id_submit, (l, 1))
        df_submit.to_csv(get_dataset_folder(dataset.dataset_id) + '/submit/submit_%s.csv' % model.round_id, index=False)

    # save model importance
    model.save_importance()

    # save predictions (eval and test set)
    pickle.dump([y_pred_eval, y_pred_test, y_pred_submit],
                open(get_dataset_folder(dataset.dataset_id) + '/predict/%s.pkl' % model.round_id, 'wb'))

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


def __cv(model, dataset, ds, pipeline, threshold):
        # performs a cross validation on cv_folds, and predict also on X_test
        y_pred_eval, y_pred_test, y_pred_submit = [], [], []
        for i, (train_index, eval_index) in enumerate(ds.cv_folds):
            if i == 0 and model.early_stopping:
                print('early stopping round')
                # with early stopping, we perform an initial round to get number of rounds
                X1, y1 = __resample(pipeline, ds.X_train[train_index], ds.y_train[train_index])
                model.fit_early_stopping(X1, y1, ds.X_train[eval_index], ds.y_train[eval_index])

                if threshold != 0:
                    # test outlier (i.e. exceeds threshold)
                    y_pred = model.predict(ds.X[eval_index])
                    score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
                    print('early stopping score: %.5f' % score)
                    if score > threshold:
                        print('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
                        time.sleep(10)
                        return True, 0, 0, 0

            # then train on train set and predict on eval set
            X1, y1 = __resample(pipeline, ds.X_train[train_index], ds.y_train[train_index])
            model.fit(X1, y1)
            y_pred = model.predict(ds.X_train[eval_index])

            if threshold != 0:
                # test outlier:
                score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
                print('fold %d score: %.5f' % (i, score))
                if score > threshold:
                    print('%dth round found outlier: %.5f with threshold %.5f' % (i, score, threshold))
                    time.sleep(10)
                    return True, 0, 0, 0

            y_pred_eval.append(y_pred)

            # we also predict on test & submit set (to be averaged later)
            y_pred_test.append(model.predict(ds.X_test))

        if dataset.mode == 'standard':
            # train on complete train set
            X1, y1 = __resample(pipeline, ds.X_train, ds.y_train)
            model.fit(X1, y1)
            y_pred_test = model.predict(ds.X_test)
        else:
            # train on complete X y set
            X1, y1 = __resample(pipeline, ds.X, ds.y)
            model.fit(X1, y1)
            if dataset.mode == 'competition':
                y_pred_submit = model.predict(ds.X_submit)
                # test = mean of y_pred_test on multiple folds
                y_pred_test = np.mean(y_pred_test, axis=0)
            else:
                y_pred_test = model.predict(ds.X_test)

        return False, y_pred_eval, y_pred_test, y_pred_submit


def __resample(pipeline, X, y):
    # apply resampling steps in pipeline
    for ref, category, name, params in pipeline:
        if category == 'sampling':
            solution = pp_solutions_map[ref]
            p_class = solution.process
            process = p_class(params)
            print('executing process', category, name, params)
            return process.fit_sample(X, y)
    print('Warning: resampling not found')
    return X, y


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
    preds_submit = [x[2] for x in preds]

    return EnsemblePool(round_ids, model_names, preds_eval, preds_test, preds_submit)


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
