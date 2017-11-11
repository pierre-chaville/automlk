import pickle
from automlk.dataset import get_dataset_list, create_graph_data
from automlk.graphs import *
from automlk.worker import get_search_rounds
from automlk.models import get_pred_eval_test

"""
module specifically designed to update feature round graphs after new version
"""

for dataset in get_dataset_list():
    print('-'*60)
    print(dataset.name)
    ds = pickle.load(open(get_dataset_folder(dataset.dataset_id) + '/data/eval_set.pkl', 'rb'))

    for msg_search in get_search_rounds(dataset.dataset_id).to_dict(orient='records'):
        try:
            print('round:', msg_search['round_id'])
            y_pred_eval, y_pred_test, y_pred_submit = get_pred_eval_test(dataset.dataset_id, msg_search['round_id'])

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

        except:
            print('error on graph update')
            print(msg_search)
