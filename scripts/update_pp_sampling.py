from automlk.dataset import get_dataset_list, create_dataset_sets
from automlk.store import list_key_store, set_key_store

"""
module specifically designed to update pp sampling results
"""

i = 0
for dt in get_dataset_list():
    if dt.status != 'created' and dt.problem_type == 'classification':
        print(dt.name)
        results = list_key_store('dataset:%s:rounds' % dt.dataset_id)
        set_key_store('dataset:%s:rounds_backup' % dt.dataset_id, results)
        flag = False
        for r in results:
            pipeline = r['pipeline']
            # print(pipeline)
            sampling = [p for p in pipeline if p[1] == 'sampling']
            if len(sampling) == 0:
                print('missing sampling step')
                r['pipeline'].append(('SP_PASS', 'sampling', 'No re-sampling', {}))
                i += 1
                flag = True

        if flag:
            set_key_store('dataset:%s:rounds' % dt.dataset_id, results)

print('modified pipelines:', i)




