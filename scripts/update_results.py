from automlk.dataset import get_dataset_list
from automlk.store import get_key_store
from automlk.solutions_pp import pp_solutions_map

"""
update pipeline codes in results with new pre-processing codes
"""

missing = []

for dt in get_dataset_list():
    print('-'*60)
    print(dt.name)

    res = get_key_store('dataset:%s:rounds' % dt.dataset_id)
    if res is not None:
        for r in res:
            for p in r['pipeline']:
                if p[0] not in pp_solutions_map:
                    if p[0] not in missing:
                        missing.append(p[0])
                        print(p[0])
        #print(res[0]['pipeline'])

print(missing)
