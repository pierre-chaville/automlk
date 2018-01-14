from automlk.dataset import get_dataset_list
from automlk.prepare import prepare_dataset_sets

"""
module specifically designed to update train/test set preparation
"""

for dt in get_dataset_list():
    print(dt.name)
    prepare_dataset_sets(dt)
