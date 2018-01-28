from .store import get_key_store, exists_key_store, set_key_store, incr_key_store, rpush_key_store


def get_folder_list():
    """
    returns the list of folders

    :return:
    """
    if exists_key_store('folders:list'):
        return get_key_store('folders:list')
    else:
        # initialize folder list with root (All)
        set_key_store('folders:counter', 0)
        rpush_key_store('folders:list', {'id': 0, 'parent': -1, 'name': 'All'})
        return get_key_store('folders:list')


def create_folder(id_parent, name):
    """
    creates the folder under parent id

    :param id_parent: id
    :return:
    """
    id_folder = incr_key_store('folders:counter')
    rpush_key_store('folders:list', {'id': id_folder, 'parent': id_parent, 'name': name})
    return id_folder


def update_folder(id_folder, id_parent, name):
    """
    update the folder

    :param id_folder: id
    :param id_parent: parent id
    :param name: new name of the folder
    :return:
    """
    ll = get_key_store('folders:list')
    for l in ll:
        if l['id'] == id_folder:
            l['name'] = name
            l['parent'] = id_parent
    set_key_store('folders:list', ll)


def delete_folder(id_folder):
    """
    deletes the folder

    :param id_folder: id
    :return:
    """
    set_key_store('folders:list', [l for l in get_key_store('folders:list') if l['id'] != id_folder])


def has_subfolders(folder_id):
    """
    indicates if a folder has sub folders

    :param folder:
    :return:
    """
    for f in get_folder_list():
        if f['parent'] == folder_id:
            return True
    return False