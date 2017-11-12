from automlk.store import get_key_store, list_key_store, lpush_key_store
from automlk.config import *

dataset_id = '42'
round_id = 1

results = list_key_store('dataset:%s:rounds' % dataset_id)

for r in results:
    if r['round_id'] == round_id:
        # print(r)
        # reinject this round in another dataset experiment
        msg_search = {'dataset_id': '47', 'round_id': 107, 'solution': r['solution'], 'level': 1,
         'ensemble_depth': 0, 'model_name': r['model_name'], 'model_params': r['model_params'], 'pipeline': r['pipeline'],
         'threshold': 0, 'time_limit': 10000}

        print('sending %s' % msg_search)
        lpush_key_store(SEARCH_QUEUE, msg_search)
