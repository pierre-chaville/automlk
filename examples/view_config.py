from automlk.dataset import get_data_folder, get_dataset_list


print('data folder:', get_data_folder())
print('list of datasets:', [d.name for d in get_dataset_list()])
