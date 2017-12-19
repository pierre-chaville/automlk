from copy import deepcopy
from .config import *
from .context import HyperContext, XySet
from .dataset import get_dataset
from .graphs import graph_histogram_regression, graph_histogram_classification, graph_predict_regression, graph_predict_classification
from .solutions import *
from .monitor import *
from .solutions_pp import pp_solutions_map

log = logging.getLogger(__name__)


def launch_worker():
    """
    periodically pool the receiver queue for a search job

    :return:
    """
    # check version
    if not check_installed_version():
        update_version()
        exit()

    init_timer_worker()
    msg_search = ''
    while True:
        #try:
        # poll queue
        msg_search = brpop_key_store('controller:search_queue')
        heart_beep('worker', msg_search)
        if msg_search != None:
            log.info('received %s' % msg_search)
            msg_search = {**msg_search, **{'start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                           'host_name': socket.gethostname()}}
            start_timer_worker(msg_search['time_limit'])
            job_search(msg_search)
            stop_timer_worker()
        """
        except KeyboardInterrupt:
            log.info('Keyboard interrupt: exiting')
            abort_timer_worker()
            exit()
        except Exception as e:
            log.error(e)
            with open(get_data_folder() + '/errors.txt', 'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + str(msg_search) + '\n')
                f.write(str(e) + '\n')
                f.write('-'*80 + '\n')
        """

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
        pool = None
    else:
        msg_search['duration_process'] = 0
        ds = ds_ini
        final_pipeline = []
        pool = __get_pool_models(dataset, msg_search['ensemble_depth'])
        context.feature_names = __get_pool_features(dataset, pool)

    solution = model_solutions_map[msg_search['solution']]
    if solution.is_wrapper:
        model = solution.model(dataset, context, msg_search['model_params'])
    else:
        model = solution.model(**msg_search['model_params'])

    msg_search['model_class'] = model.__class__.__name__
    __search(dataset, context, solution, model, msg_search, ds, pool, final_pipeline)


def __pre_processing(context, pipeline, ds):
    # performs the different pre-processing steps
    context.pipeline = pipeline
    for ref, category, name, params in pipeline:
        if category != 'sampling':
            solution = pp_solutions_map[ref]
            p_class = solution.process
            process = p_class(context, params)
            log.info('executing process %s %s %s' % (category, name, process.params))
            ds.X_train = process.fit_transform(ds.X_train, ds.y_train)
            ds.X_test = process.transform(ds.X_test)
            ds.X = process.transform(ds.X)
            if len(ds.X_submit) > 0:
                ds.X_submit = process.transform(ds.X_submit)
            log.info('-> %d features' % len(context.feature_names))
    final_pipeline = [p for p in pipeline if p[1] == 'sampling']
    log.info('final pipeline %s' % final_pipeline)
    return context, ds, final_pipeline


def __search(dataset, context, solution, model, msg_search, ds, pool, pipeline):
    log.info('optimizing with %s, params: %s' % (solution.name, msg_search['model_params']))
    # fit, test & score
    t_start = time.time()
    round_id = msg_search['round_id']
    level = msg_search['level']
    if level == 2:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = __cv_pool(solution, model, dataset, pool, ds, msg_search['threshold'],
                                                                                        msg_search['ensemble_depth'])
    else:
        outlier, y_pred_eval_list, y_pred_test, y_pred_submit = __cv(solution, model, dataset, ds, pipeline,
                                                                     msg_search['threshold'])
    if hasattr(model, 'num_rounds'):
        msg_search['num_rounds'] = model.num_rounds
    else:
        msg_search['num_rounds'] = None

    # check outlier
    if outlier:
        log.info('outlier, skipping this round')
        return

    # y_pred_eval as concat of folds
    y_pred_eval = np.concatenate(y_pred_eval_list)

    # reindex eval to be aligned with y
    y_pred_eval[ds.i_eval] = y_pred_eval.copy()

    # generate submit file
    if dataset.filename_submit != '':
        ls = len(ds.id_submit)
        #if dataset.problem_type == 'regression':
        if np.shape(y_pred_submit)[1] == 1:
            submit = np.concatenate((np.reshape(ds.id_submit, (ls, 1)), np.reshape(y_pred_submit, (ls, 1))), axis=1)
        else:
            submit = np.concatenate((np.reshape(ds.id_submit, (ls, 1)), np.reshape(y_pred_submit[:, 1], (ls, 1))), axis=1)
        df_submit = pd.DataFrame(submit)
        df_submit.columns = [dataset.col_submit, dataset.y_col]
        # allocate id column to avoid type conversion (to float)
        df_submit[dataset.col_submit] = np.reshape(ds.id_submit, (ls, 1))
        df_submit.to_csv(get_dataset_folder(dataset.dataset_id) + '/submit/submit_%s.csv' % round_id, index=False)

    # save model importance
    if level == 2 and solution.is_wrapper:
        __save_importance(model.model, dataset, context, round_id)
    else:
        __save_importance(model, dataset, context, round_id)

    # save predictions (eval and test set)
    pickle.dump([y_pred_eval, y_pred_test, y_pred_submit],
                open(get_dataset_folder(dataset.dataset_id) + '/predict/%s.pkl' % round_id, 'wb'))

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


def __cv(solution, model, dataset, ds, pipeline, threshold):
        # performs a cross validation on cv_folds, and predict also on X_test
        y_pred_eval, y_pred_test, y_pred_submit = [], [], []
        for i, (train_index, eval_index) in enumerate(ds.cv_folds):
            if i == 0 and solution.early_stopping:
                log.info('early stopping round')
                # with early stopping, we perform an initial round to get number of rounds
                X1, y1 = __resample(pipeline, ds.X_train[train_index], ds.y_train[train_index])
                model.fit_early_stopping(X1, y1, ds.X_train[eval_index], ds.y_train[eval_index])

                if threshold != 0:
                    # test outlier (i.e. exceeds threshold)
                    y_pred = __predict(solution, model, ds.X[eval_index])
                    score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
                    if score > threshold:
                        log.info('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
                        time.sleep(10)
                        return True, 0, 0, 0

            # then train on train set and predict on eval set
            X1, y1 = __resample(pipeline, ds.X_train[train_index], ds.y_train[train_index])
            model.fit(X1, y1)
            y_pred = __predict(solution, model, ds.X_train[eval_index])

            if threshold != 0:
                # test outlier:
                score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
                if score > threshold:
                    log.info('%dth round found outlier: %.5f with threshold %.5f' % (i, score, threshold))
                    time.sleep(10)
                    return True, 0, 0, 0

            y_pred_eval.append(y_pred)

            # we also predict on test & submit set (to be averaged later)
            y_pred_test.append(__predict(solution, model, ds.X_test))

        if dataset.mode == 'standard':
            # train on complete train set
            X1, y1 = __resample(pipeline, ds.X_train, ds.y_train)
            model.fit(X1, y1)
            y_pred_test = __predict(solution, model, ds.X_test)
        else:
            # train on complete X y set
            X1, y1 = __resample(pipeline, ds.X, ds.y)
            model.fit(X1, y1)
            if dataset.mode == 'competition':
                y_pred_submit = __predict(solution, model, ds.X_submit)
                # test = mean of y_pred_test on multiple folds
                y_pred_test = np.mean(y_pred_test, axis=0)
            else:
                y_pred_test = __predict(solution, model, ds.X_test)

        return False, y_pred_eval, y_pred_test, y_pred_submit


def __cv_pool(solution, model, dataset, pool, ds, threshold, depth):
    y_pred_eval, y_pred_test, y_pred_submit = [], [], []

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
            X_train = p_eval
            X_test = p_test
            if dataset.mode == 'competition':
                X_submit = p_submit
        else:
            # stack vertically the predictions
            X_train = np.concatenate((X_train, p_eval), axis=1)
            X_test = np.concatenate((X_test, p_test), axis=1)
            if dataset.mode == 'competition':
                X_submit = np.concatenate((X_submit, p_submit), axis=1)

    for i, (train_index, eval_index) in enumerate(ds.cv_folds):
        log.info('fold %d' % i)

        if i == 0 and solution.early_stopping:
            # with early stopping, we perform an initial round to get number of rounds
            log.info('fit early stopping')
            model.fit_early_stopping(X_train[train_index], ds.y_train[train_index], X_train[eval_index], ds.y_train[eval_index])
            y_pred = __predict(solution, model, X_train[eval_index])
            score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
            if threshold != 0 and score > threshold:
                log.info('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
                time.sleep(10)
                return True, 0, 0, 0

        # train on X_train
        model.fit(X_train[train_index], ds.y_train[train_index])
        y_pred = __predict(solution, model, X_train[eval_index])
        y_pred_eval.append(y_pred)
        y_pred_test.append(__predict(solution, model, X_test))
        score = dataset.evaluate_metric(ds.y_train[eval_index], y_pred)
        if threshold != 0 and score > threshold:
            log.info('found outlier: %.5f with threshold %.5f' % (score, threshold))
            time.sleep(10)
            return True, 0, 0, 0

    if dataset.mode == 'standard':
        # train on complete train set
        model.fit(X_train, ds.y_train)
        y_pred_test = __predict(solution, model, X_test)
    else:
        # train on complete X y set
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((ds.y_train, ds.y_test), axis=0)
        model.fit(X, y)
        if dataset.mode == 'competition':
            y_pred_submit = __predict(solution, model, X_submit)
            # test = mean of y_pred_test on multiple folds
            y_pred_test = np.mean(y_pred_test, axis=0)
        else:
            y_pred_test = __predict(solution, model, X_test)

    return False, y_pred_eval, y_pred_test, y_pred_submit


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


def __save_importance(model, dataset, context, round_id):
    # saves feature importance (as a dataframe)
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame(context.feature_names)
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

    log.info('length of pool: %d for ensemble of depth %d' % (len(round_ids), depth))

    # retrieves predictions
    preds = [get_pred_eval_test(dataset.dataset_id, round_id) for round_id in round_ids]

    # exclude predictions with nan
    excluded = [i for i, x in enumerate(preds) if not(np.max(x[0]) == np.max(x[0]))]

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



def get_search_rounds(dataset_id):
    """
    get all the results of the search with preprocessing and models

    :param dataset_id: id of the dataset
    :return: results of the search as a dataframe
    """
    results = list_key_store('dataset:%s:rounds' % dataset_id)
    return pd.DataFrame(results)
