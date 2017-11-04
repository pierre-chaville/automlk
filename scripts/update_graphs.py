from automlk.dataset import get_dataset_list, create_graph_data
from automlk.store import set_key_store

"""
module specifically designed to update feature graphs after new version
"""

for dt in get_dataset_list():
    print('-'*60)
    print(dt.name)
    create_graph_data(dt.dataset_id)
