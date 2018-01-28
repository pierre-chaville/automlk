from automlk.dataset import get_dataset_list, update_dataset
from automlk.store import get_key_store
from automlk.solutions_pp import pp_solutions_map
from automlk.folders import create_folder, get_folder_list

"""
upgrade folders

"""


folder_list = get_folder_list()
folders = [f['name'] for f in folder_list]

for dt in get_dataset_list():
    print('-'*60)
    print(dt.name)

    res = get_key_store('dataset:%s' % dt.dataset_id)
    name = res['init_data']['domain']
    if name not in folders:
        folders.append(name)
        # create this folder
        folder_id = create_folder(0, name)

        folder_list = get_folder_list()
        folders = [f['name'] for f in folder_list]
    else:
        folder_id = 0
        for f in folder_list:
            if f['name'] == name:
                folder_id = f['id']
                break

    # update dataset
    update_dataset(dt.dataset_id, dt.name, folder_id, dt.description, dt.source, dt.url)

print(folders)

