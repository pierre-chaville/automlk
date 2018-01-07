import datetime
from .store import *


def create_textset(name, description, source, url, filename):
    """
    creates a text set object

    :param name: name of the text set
    :param filename: filename of the data
    :return: a textset object
    """
    # create object and control data
    creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ts = TextSet(0, name, description, source, url, filename, creation_date)

    # save and create objects related to the textset
    ts.textset_id = str(incr_key_store('textset:counter'))
    set_key_store('textset:%s:status' % ts.textset_id, 'created')
    rpush_key_store('textset:list', ts.textset_id)

    ts.finalize_creation()
    ts.save()
    return ts


def get_textset(textset_id):
    """
    retrieves a textset object from its id
    :param textset_id: id
    :return: textset object
    """
    d = get_key_store('textset:%s' % textset_id)
    ds = TextSet(**d['init_data'])
    ds.load(d['load_data'])
    ds.status = get_key_store('textset:%s:status' % textset_id)
    return ds


def get_textset_status(textset_id):
    """
    retrieves the status of a textset from its id
    :param textset_id: id
    :return: status (string)
    """
    return get_key_store('textset:%s:status' % textset_id)


def get_textset_list():
    """
    get the list of all textsets

    :return: list of textsets objects or empty list if error (eg. redis or environment not set)
    """
    #try:
    return [get_textset(textset_id) for textset_id in get_textset_ids()]
    #except:
    #    return []


def get_textset_ids():
    """
    get the list of ids all textsets

    :return: list of ids
    """
    return list_key_store('textset:list')


def update_textset(textset_id, name, description, source, url):
    """
    update specific fields of the textset

    :param textset_id: id of the textset
    :param name: new name of the textset
    :param description: new description of the textset
    :param source: source of the textset
    :param url: url of the textset
    :return:
    """
    ts = get_textset(textset_id)
    ts.name = name
    ts.description = description
    ts.source = source
    ts.url = url
    ts.save()


def reset_textset(textset_id):
    """
    reset the results

    :param textset_id: id
    :return:
    """
    # removes entries
    set_key_store('textset:%s:status' % textset_id, 'created')


def delete_textset(textset_id):
    """
    deletes a textset and the results

    :param textset_id: id
    :return:
    """
    # removes entries
    del_key_store('textset:%s:status' % textset_id)
    lrem_key_store('textset:list', textset_id)
    # delete file
    os.remove(get_data_folder() + '/texts' + '/' + str(textset_id) + '.txt')


class TextSet(object):
    def __init__(self, textset_id, name, description, source, url, filename, creation_date):
        self.textset_id = textset_id
        self.name = name
        self.description = description
        self.url = url
        self.source = source  # url or file id

        if filename == '':
            raise ValueError('filename cannot be empty')

        self.filename = filename
        ext = filename.split('.')[-1].lower()
        if ext not in ['txt']:
            raise TypeError('unknown text format: use txt')

        if not os.path.exists(filename):
            raise ValueError('file %s not found' % filename)

        self.size = 0
        self.lines = 0
        self.creation_date = creation_date

    def finalize_creation(self):
        # import text
        with open(self.filename, 'r') as f:
            txt = f.readlines()
        self.size = sum([len(s) for s in txt])
        self.lines = len(txt)
        folder = get_data_folder() + '/texts'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + '/' + str(self.textset_id) + '.txt', 'w') as f:
            for s in txt:
                f.write(s + '\n')

    def save(self):
        store = {'init_data': {'textset_id': self.textset_id, 'name': self.name, 'description': self.description,
                               'source': self.source, 'url': self.url, 'filename': self.filename,
                               'creation_date': self.creation_date},
                 'load_data': {'size': self.size, 'lines': self.lines}}
        set_key_store('textset:%s' % self.textset_id, store)

    def load(self, store):
        # reload data from json
        for k in store.keys():
            setattr(self, k, store[k])

