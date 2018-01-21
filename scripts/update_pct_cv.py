from automlk.dataset import get_dataset_list
from automlk.store import get_key_store, set_key_store, exists_key_store

"""
update results with pct and cv
"""

missing = []

for dt in get_dataset_list():

    res = get_key_store('dataset:%s:rounds' % dt.dataset_id)
    flag = False
    if res is not None:
        for r in res:
            if 'pct' not in r.keys():
                r['pct'] = 1.
                flag = True
            if 'cv' not in r.keys():
                r['cv'] = True
                flag = True
            if 'mode' not in r.keys():
                r['mode'] = 'search'
                flag = True
    # update store
    if flag:
        set_key_store('dataset:%s:rounds' % dt.dataset_id, res)
        print('updating results:', dt.name)

    key = 'dataset:%s:best' % dt.dataset_id
    if exists_key_store(key):
        res = get_key_store(key)
        flag = False
        if res is not None:
            for r in res:
                if 'pct' not in r.keys():
                    r['pct'] = 1.
                    flag = True
                if 'cv' not in r.keys():
                    r['cv'] = True
                    flag = True
                if 'mode' not in r.keys():
                    r['mode'] = 'search'
                    flag = True
        # update store
        if flag:
            set_key_store(key, res)
            print('updating best results:', dt.name)

    key = 'dataset:%s:best_pp' % dt.dataset_id
    if exists_key_store(key):
        res = get_key_store(key)
        flag = False
        if res is not None:
            for cat, rr in res:
                for r in rr:
                    if 'pct' not in r.keys():
                        r['pct'] = 1.
                        flag = True
                    if 'cv' not in r.keys():
                        r['cv'] = True
                        flag = True
                    if 'mode' not in r.keys():
                        r['mode'] = 'search'
                        flag = True
        # update store
        if flag:
            set_key_store(key, res)
            print('updating best results pp:', dt.name)
