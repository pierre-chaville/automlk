import logging
import uuid
from .config import *
from .store import *
from .context import get_uploads_folder
from .dataset import get_dataset_ids, get_dataset
from .textset import create_textset, get_textset_status
from .solutions import *
from .solutions_pp import *
from .worker import get_search_rounds
from .monitor import heart_beep, set_installed_version
from .graphs import graph_history_search
from .specific import *
from .prepare import prepare_dataset_sets

PATIENCE = 500  # number of equivalent results to wait before stop
ROUNDS_MAX = 5000  # number max of rounds before stop

ROUND_START_ENSEMBLE = 200  # number of rounds to start ensembles
RATIO_L1_L2 = 2  # number of L1 rounds compared to L2 rounds

RATIO_ROUNDS = 10  # number rounds -> equivalent in number of results (only a fraction can pass threshold)
RATIO_START_THRESHOLD = 100  # number of rounds to start applying a threshold
RATIO_MIN = 2  # minimum models to include in threshold (should be > 1)
RATIO_THRESHOLD_MAX = 50  # maximum % of models to include in threshold (should be > 1)
RATIO_THRESHOLD_SLOPE = 10  # % of models decrease per 50 results to include in threshold (should be > 1)

log = logging.getLogger(__name__)


def launch_controller():
    """
    controller process: manages the search strategy and send instruction to workers
    :return:
    """
    # update installed version on disk to synchronize automatically the version of the controllers
    set_installed_version()

    i_dataset = -1
    # controls the optimization rounds and sends instructions to the workers
    while True:
        # check the list of datasets to search
        active = [id for id in get_dataset_ids() if get_key_store('dataset:%s:status' % id) == 'searching']

        if len(active) == 0:
            heart_beep('controller', {})
            time.sleep(1)
        else:
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
                if msg_search != {}:
                    log.info('sending %s' % msg_search)
                    lpush_key_store(SEARCH_QUEUE, msg_search)
                    heart_beep('controller', msg_search)

        # then read the duplicate ROUND queue
        while llen_key_store(DUPLICATE_QUEUE) > 0:
            msg = brpop_key_store(DUPLICATE_QUEUE)
            msg_search = __duplicate_search_round(msg['round'], msg['dataset'])

            # send queue the next job to do
            log.info('sending %s' % msg_search)
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

    # generate round id
    round_id = incr_key_store('dataset:%s:round_counter' % dataset_id) - 1
    if round_id == 0:
        # first launch: create train & eval & test set
        prepare_dataset_sets(dataset)

    # check if text set is completed
    if len(dataset.text_cols) > 0:
        if not __check_text_sets(dataset):
            return {}

    # initialize search parameters
    level = 1
    default_mode = False
    ensemble_depth = 0
    threshold = 0
    df = pd.DataFrame()

    # generate model and model params
    l1_choices = __get_model_class_list(dataset, 1)

    if round_id < len(l1_choices):
        # default mode, level 1
        default_mode = True
        round_id_l1 = round_id
        i_choice = round_id_l1 % len(l1_choices)
        ref = l1_choices[i_choice]
        solution = model_solutions_map[ref]
        params = solution.default_params
    else:
        # get search history
        df = get_search_rounds(dataset_id)
        # find threshold
        threshold = __focus_threshold(df, round_id)
        level, round_id_l1, round_id_l2 = __get_round_ids(round_id)
        log.info('get_ids: round_id:%d, level=%d, l1=%d, l2=%d' % (round_id, level, round_id_l1, round_id_l2))

        if level == 2:
            l2_choices = __get_model_class_list(dataset, 2)
            i_choice = round_id_l2 % len(l2_choices)
            ref = l2_choices[i_choice]
            ensemble_depth = random.randint(1, 3)
        else:
            # focus on models > threshold
            if threshold != 0:
                l1_choices = [x[0] for x in __get_list_best_models(df) if x[1] <= threshold]
                log.info('focusing on %s' % l1_choices)

            i_choice = round_id_l1 % len(l1_choices)
            ref = l1_choices[i_choice]

        solution = model_solutions_map[ref]
        params = get_random_params(solution.space_params)

    # check params
    rule = solution.rule_params
    if rule:
        params = rule(dataset, default_mode, params)

    # generate pre-processing pipeline and pre-processing params
    if level == 1:
        pipeline = __get_pipeline(dataset, solution, default_mode, round_id_l1, df, threshold)
    else:
        pipeline = [('FR-PASS', 'feature', 'No Feature selection', {})]

    # generate search message
    return {'dataset_id': dataset.dataset_id, 'round_id': round_id, 'solution': solution.ref, 'level': level,
            'ensemble_depth': ensemble_depth, 'model_name': solution.name, 'model_params': params, 'pipeline': pipeline,
            'threshold': threshold, 'time_limit': __time_limit(dataset)}  # 'percent_data': 100, 'cv': True}


def __prepare_text_sets(dataset):
    """
    generates unsupervised text sets for the datatset

    :param dataset: dataset object
    :return:
    """
    for f in dataset.features:
        if f.name in dataset.text_cols and f.text_ref == '':
            col = f.name
            log.info('creating textset for column %s' % col)
            # generate file for text set
            filename = get_uploads_folder() + '/' + str(uuid.uuid4()) + '.txt'

            df = dataset.get_data('train')
            with open(filename, 'w') as f:
                for line in df[col].values:
                    f.write(line + '\n')

            if dataset.mode == 'benchmark':
                # add also text from test
                df = dataset.get_data('test')
                with open(filename, 'a') as f:
                    for line in df[col].values:
                        f.write(line + '\n')
            elif dataset.mode == 'competition':
                df = dataset.get_data('submit')
                with open(filename, 'a') as f:
                    for line in df[col].values:
                        f.write(line + '\n')

            # create text set
            ts = create_textset(name=dataset.name + '/' + col,
                                description='generated automatically for dataset: %s (id=%s) and column : %s' % (
                                dataset.name, dataset.dataset_id, col),
                                source='train and test set', url='',
                                filename=filename)

            # update ref in dataset features
            for f in dataset.features:
                if f.name == col:
                    f.text_ref = ts.textset_id

            log.info('created textset id : %s' % ts.textset_id)

    # then save dataset
    dataset.save(dataset.dataset_id)


def __check_text_sets(dataset):
    """
    check if all text sets (references only) models are completed

    :param dataset: dataset object
    :return:
    """
    for f in dataset.features:
        if f.name in dataset.text_cols and f.text_ref != '':
            if get_textset_status(f.text_ref) != 'completed':
                return False
    return True


def __duplicate_search_round(round, dataset_id):
    """
    apply the parameters of the round to search in the target dataset

    :param round: round parameters
    :param dataset_id: id of the target dataset
    :return:
    """
    dataset = get_dataset(dataset_id)

    # generate round id
    round_id = incr_key_store('dataset:%s:round_counter' % dataset_id) - 1
    if round_id == 0:
        # first launch: create train & eval & test set
        prepare_dataset_sets(dataset)

    # generate search message:
    return {'dataset_id': dataset.dataset_id, 'round_id': round_id, 'solution': round["solution"],
            'level': round["level"], 'ensemble_depth': round["ensemble_depth"], 'model_name': round["model_name"],
            'model_params': round["model_params"], 'pipeline': round["pipeline"],
            'threshold': 0, 'time_limit': __time_limit(dataset)}


def __get_round_ids(round_id):
    # find round ids for each level; return = level, round_id_l1, round_id_l2

    if round_id < ROUND_START_ENSEMBLE:
        return 1, round_id, -1
    round_id_l1 = -1
    round_id_l2 = -1
    for i in range(round_id + 1):
        if i >= ROUND_START_ENSEMBLE and i % RATIO_L1_L2 == 0:
            round_id_l2 += 1
        else:
            round_id_l1 += 1
    if round_id >= ROUND_START_ENSEMBLE and round_id % RATIO_L1_L2 == 0:
        return 2, round_id_l1, round_id_l2
    else:
        return 1, round_id_l1, round_id_l2


def __focus_threshold(df, round_id):
    """
    calculates the threshold for outliers from the score history

    :param df:
    :param round_id:
    :return:
    """
    if len(df) < RATIO_START_THRESHOLD:
        return 0

    df0 = df[(df.cv_max != METRIC_NULL) & (df.level == 1)].groupby('model_name', as_index=False).min()
    scores = np.sort(df0.cv_max.values)
    if len(scores) < 5:
        return 0

    # we take into account the length of results, but also we cap the round_id (1/10):
    base = max(len(df), round_id / RATIO_ROUNDS) - RATIO_START_THRESHOLD

    # we will decrease the threshold from max % of the best scores to min%
    ratio = RATIO_THRESHOLD_MAX - base * RATIO_THRESHOLD_SLOPE / 50
    n = max(RATIO_MIN, round(len(scores) * ratio / 100))
    threshold = scores[n - 1]
    log.info('outlier threshold set at: %.5f (ratio=%.2f%%, n=%d, %d scores)' % (threshold, ratio, n, len(scores)))
    return threshold


def __time_limit(dataset):
    # determine max delay to execute search
    if dataset.n_rows < 10:
        return 900 * int(1 + dataset.n_rows)
    elif dataset.n_rows < 100:
        return 3 * 3600
    else:
        return 6 * 3600


def __get_pipeline(dataset, solution, default_mode, i_round, df, threshold):
    # generates the list of potential data pre-processing depending on the problem type
    pipeline = []
    if threshold == 0:
        best_pp = None
    else:
        best_pp = __get_list_best_pp(df, cv_max=True)

    # missing values
    if len(dataset.missing_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'missing', solution, default_mode, i_round, best_pp, threshold))

    # numerical values
    numeric_cols = [f.name for f in dataset.features if f.raw_type.startswith('float')]
    if len(numeric_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'float', solution, default_mode, i_round, best_pp, threshold))

    # date values
    date_cols = [f.name for f in dataset.features if f.col_type == 'date']
    if len(date_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'date', solution, default_mode, i_round, best_pp, threshold))

    # X pre-processing: text
    if len(dataset.text_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'text', solution, default_mode, i_round, best_pp, threshold))

    # X pre-processing: categorical
    if len(dataset.cat_cols) > 0:
        pipeline.append(__get_pp_choice(dataset, 'categorical', solution, default_mode, i_round, best_pp, threshold))

    # scaling
    pipeline.append(__get_pp_choice(dataset, 'scaling', solution, default_mode, i_round, best_pp, threshold))

    # feature selection
    pipeline.append(__get_pp_choice(dataset, 'feature', solution, default_mode, i_round, best_pp, threshold))

    # re-sampling
    if dataset.problem_type == 'classification' and dataset.sampling:
        pipeline.append(__get_pp_choice(dataset, 'sampling', solution, default_mode, i_round, best_pp, threshold))

    # clean and return the pipeline from null steps
    return [p for p in pipeline if p[0] != '']


def __get_pp_choice(dataset, category, solution, default_mode, i_round, best_pp, threshold):
    # select a solution bewteen the list of potential solutions from the category
    if threshold == 0:
        choices = __get_pp_list(dataset, category, solution, default_mode)
    else:
        choices = [x[0] for x in best_pp if (x[1] == category) and (x[2] <= threshold)]
        log.info('focusing on pre-processing for %s / %s' % (category, choices))

    if len(choices) == 0:
        return '', category, '', {}

    i_pp = i_round % len(choices)
    ref = __check_ref(choices[i_pp], category)
    s = pp_solutions_map[ref]
    if default_mode:
        params = s.default_params
    else:
        params = get_random_params(s.space_params)
    return ref, category, s.name, params


def __check_ref(ref, category):
    """
    upward compatibilty of references

    :param ref: input ref
    :param category: category of processing
    :return: updated ref
    """
    if ref == '' and category == 'float':
        return 'FL-PASS'
    return ref


def __get_pp_list(dataset, category, solution, default_mode):
    # generates the list of potential pre-processing choices in a specific category, depending on the problem type
    choices = []
    if default_mode:
        l_solutions = [pp_solutions_map[p] for p in solution.pp_default]
    else:
        l_solutions = [pp_solutions_map[p] for p in solution.pp_list]
    for s in l_solutions:
        if s.pp_type == category and dataset.n_rows < s.limit_size:
            if s.problem_type == '*' or s.problem_type == dataset.problem_type:
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

    # check if the search has been reset (round_id > round counter)
    round_counter = get_key_store('dataset:%s:round_counter' % dataset_id)
    if not isinstance(round_counter, int):
        log.info('abort round because dataset current counter is not int')
        return
    elif int(msg_result['round_id']) > int(round_counter):
        log.info('round %s skipped because greater than current counter' % msg_result['round_id'])
        return

    # update search history
    rpush_key_store('dataset:%s:rounds' % dataset_id, msg_result)

    # get search history
    df = get_search_rounds(dataset_id)

    dataset = get_dataset(dataset_id)

    # generate graphs
    best = __get_best_models(df)
    best1 = best[best.level == 1]
    best2 = best[best.level == 2]
    graph_history_search(dataset, df, best1, 1)
    graph_history_search(dataset, df, best2, 2)

    # then update search
    set_key_store('dataset:%s:results' % dataset_id, len(df))
    set_key_store('dataset:%s:best' % dataset_id, best.to_dict(orient='records'))
    set_key_store('dataset:%s:best_pp' % dataset_id, __get_best_pp(df))

    # then check patience
    if len(best2 > 0):
        last_best, last_best_id = __get_last_best(df)
        if msg_result['round_id'] - last_best_id > PATIENCE:
            log.info('patience reached for dataset %s at round %d (last best: %d): search completed' % (
                dataset_id, msg_result['round_id'], last_best_id))
            set_key_store('dataset:%s:status' % dataset_id, 'completed')


def __get_last_best(df):
    # returns last best value and its index
    best = METRIC_NULL
    round_best = -1
    for round_id, score in zip(df.round_id.values, df.cv_mean.values):
        if score < best:
            best = score
            round_id = round_id
    return best, round_id


def __get_list_best_models(df, level=1):
    # returns best models
    if len(df[df.level == level]) < 1:
        return []
    best = df[df.level == level].sort_values(by=['solution', 'cv_max']). \
        groupby('solution', as_index=False).first().sort_values(by='cv_max')
    return [(x, y) for x, y in zip(best.solution.values, best.cv_max.values)]


def __get_best_models(df):
    # get the best results per model
    if len(df) < 1:
        return pd.DataFrame()
    best = df.sort_values(by=['model_name', 'cv_mean']).groupby('model_name', as_index=False).first().sort_values(
        by='cv_mean').fillna('')
    counts = df[['model_name', 'round_id']].groupby('model_name', as_index=False).count()
    counts.columns = ['model_name', 'searches']
    # relative performance
    best['rel_score'] = abs(100 * (best.cv_mean - best.cv_mean.max()) / (best.cv_mean.max() - best.cv_mean.min()))
    return pd.merge(best, counts, on='model_name')


def __get_list_best_pp(df, cv_max=False):
    # get the best results per pre-processing
    l = []
    for cat, l_cat in __get_best_pp(df, cv_max):
        for p in l_cat:
            l.append((p['cat_ref'], cat, p['cv_max']))
    return l


def __get_best_pp(df, cv_max=False):
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
    for c in PP_CATEGORIES:
        if c in cat:
            df['cat_ref'] = df['pipeline'].map(lambda x: __select_cat(c, x)[0])
            df['cat_name'] = df['pipeline'].map(lambda x: __select_cat(c, x)[1])
            df['cat_process'] = df['pipeline'].map(lambda x: __select_cat(c, x)[2])
            df['cat_params'] = df['pipeline'].map(lambda x: __select_cat(c, x)[3])
            if cv_max:
                best = df.sort_values(by=['cat_ref', 'cv_max']).groupby('cat_ref', as_index=False).first().sort_values(
                    by='cv_max').fillna('')
            else:
                best = df.sort_values(by=['cat_ref', 'cv_mean']).groupby('cat_ref', as_index=False).first().sort_values(
                    by='cv_mean').fillna('')

            counts = df[['cat_ref', 'round_id']].groupby('cat_ref', as_index=False).count()
            counts.columns = ['cat_ref', 'searches']

            # relative performance
            if cv_max:
                best['rel_score'] = abs(
                    100 * (best.cv_max - best.cv_max.max()) / (best.cv_max.max() - best.cv_max.min()))
            else:
                best['rel_score'] = abs(
                    100 * (best.cv_mean - best.cv_mean.max()) / (best.cv_mean.max() - best.cv_mean.min()))

            all_cat.append((c, pd.merge(best, counts, on='cat_ref').to_dict(orient='records')))

    return all_cat


def __select_cat(c, pipeline):
    # select the element in the pipeline with category c
    for p in pipeline:
        if p[1] == c:
            return p
    return '', '', '', ''
