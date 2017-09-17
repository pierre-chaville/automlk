from automlk.store import *
from automlk.dataset import get_dataset_ids, get_dataset, get_dataset_list


print('dataset:counter', get_key_store('dataset:counter'))
print('dataset:list', list_key_store('dataset:list'))

print('dataset ids', get_dataset_ids())

dt = get_dataset(1)
print('dt ok')

l = get_dataset_list()

