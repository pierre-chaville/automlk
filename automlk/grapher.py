import time
import pickle
from .store import *
from .graphs import *
from .monitor import heart_beep
from .dataset import get_dataset_list, get_dataset
from .specific import get_feature_engineering, apply_feature_engineering


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(module)s %(lineno)3d] %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/grapher.log'),
                        logging.StreamHandler()
                    ])

logging.info('starting grapher')


def grapher_loop():
    """
    periodically pool the grapher queue for a job

    :return: 
    """
    while True:
        heart_beep('grapher', '')
        # check the list of datasets
        for dt in get_dataset_list():
            if dt.status != 'created' and not dt.grapher:
                heart_beep('grapher', {'dataset_id': dt.dataset_id, 'dataset_name': dt.name})
                logging.info('grapher on dataset: %s' % dt.dataset_id)
                create_graph_data(dt.dataset_id)
        time.sleep(60)


def create_graph_data(dataset_id):
    """
    creates the graphs for each column feature of the dataset

    :param dataset_id: dataset id
    :return:
    """
    dataset = get_dataset(dataset_id)
    df = dataset.get_data()

    # apply feature engineering (if any)
    fe = get_feature_engineering(dataset.dataset_id)
    data_train = dataset.get_data()
    if fe != '':
        df = apply_feature_engineering(dataset.dataset_id, df)

    # create a sample set
    pickle.dump(df.head(20), open(get_dataset_folder(dataset_id) + '/data/sample.pkl', 'wb'))

    # fillna to avoid issues
    for f in dataset.features:
        if f.col_type == 'numerical':
            df[f.name].fillna(0, inplace=True)
        else:
            df[f.name].fillna('', inplace=True)
            df[f.name] = df[f.name].map(str)

    # create graph of target distrubution and correlations
    graph_histogram(dataset_id, dataset.y_col, dataset.is_y_categorical, df[dataset.y_col].values)
    graph_correl_features(dataset, df.copy())

    # create graphs for all features
    for f in dataset.features:
        if f.to_keep and f.name != dataset.y_col:
            log.info('creating graph %s for dataset:%s' % (f.name, dataset_id))
            if f.col_type == 'numerical' and dataset.problem_type == 'regression':
                graph_regression_numerical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'categorical' and dataset.problem_type == 'regression':
                graph_regression_categorical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'numerical' and dataset.problem_type == 'classification':
                graph_classification_numerical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'categorical' and dataset.problem_type == 'classification':
                graph_classification_categorical(dataset_id, df, f.name, dataset.y_col)
            elif f.col_type == 'text':
                graph_text(dataset_id, df, f.name)

    # update status grapher
    set_key_store('dataset:%s:grapher' % dataset_id, True)

