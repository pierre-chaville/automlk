from automlk.dataset import get_dataset_list, create_dataset_sets

"""
module specifically designed to update Xyset after new version
"""

for dt in get_dataset_list():
    if dt.status != 'created':
        print(dt.name)
        # update X y set
        create_dataset_sets(dt)
