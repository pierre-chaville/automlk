import pickle
from .monitor import *
from .textset import *
from .utils.text_encoders import *

log = logging.getLogger(__name__)


def get_textset_w2v(textset_id, model, size):
    """
    get the model trained on textset with the size

    :param textset_id: id of the textset
    :param model: model (w2v / d2v)
    :param size: dimension (100 or 200)
    :return: model
    """
    if model not in ['w2v', 'd2v']:
        raise ValueError('model must be w2v or d2v')
    if size not in [100, 200]:
        raise ValueError('size must be 100 or 200')
    textset = get_textset(textset_id)

    #if textset.status != 'created':
    #    raise ValueError('textset status must be created')

    return pickle.load(open(get_data_folder() + '/texts/w2v_%d_%d.pkl' % (textset_id, size), 'rb'))


def launch_worker_text():
    """
    periodically pool the receiver queue for a search job

    :return:
    """
    msg_search = ''
    while True:
        # poll queue
        textset_id = brpop_key_store('worker_text:search_queue')
        heart_beep('worker_text', str(textset_id))
        if textset_id != None:
            log.info('searching textset %d' % textset_id)

            # read textset
            textset = get_textset(textset_id)
            set_key_store('textset:%s:status' % textset_id, 'searching')

            # import text
            with open(textset.filename, 'r') as f:
                text = f.readlines()

            # calculate models
            for size in [100, 200]:
                pickle.dump(__search_word2vec(text, size), open(get_data_folder() + '/texts/w2v_%d_%d.pkl' % (textset_id, size), 'wb'))
                pickle.dump(__search_doc2vec(text, size),
                            open(get_data_folder() + '/texts/d2v_%d_%d.pkl' % (textset_id, size), 'wb'))

            # update status to completed
            set_key_store('textset:%s:status' % textset_id, 'completed')


def __search_word2vec(text, size):
    """
    calculate vectors with word2vec

    :param text: list of strings
    :param size: dimension of the vectors
    :return: model
    """
    log.info('searching word2vec model with size %d' % size)
    return model_word2vec(text, {'size': size, 'iter': 50, 'window': 11, 'min_count': 5, 'sg': 0, 'workers': 8})


def __search_doc2vec(text, size):
    """
    calculate vectors with word2vec

    :param text: list of strings
    :param size: dimension of the vectors
    :return: model
    """
    log.info('searching doc2vec model with size %d' % size)
    return model_doc2vec(text, {'size': size, 'iter': 100, 'window': 11, 'min_count': 5, 'dm': 0, 'workers': 8})