import pickle
from .monitor import *
from .context import text_model_filename
from .textset import *
from .utils.text_encoders import *
from .spaces.process import space_textset_bow, space_textset_w2v, space_textset_d2v


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
    while True:
        # check the list of datasets
        for ts in get_textset_list():
            if ts.status != 'completed':
                log.info('searching textset %s' % ts.textset_id)

                # read textset
                textset = get_textset(ts.textset_id)
                set_key_store('textset:%s:status' % ts.textset_id, 'searching')

                # import text
                with open(textset.filename, 'r') as f:
                    text = f.readlines()

                # calculate models
                for conf in space_textset_bow:
                    pickle.dump(model_bow(text, conf), open(text_model_filename(ts.textset_id, 'bow', conf), 'wb'))

                for conf in space_textset_w2v:
                    pickle.dump(model_word2vec(text, conf), open(text_model_filename(ts.textset_id, 'w2v', conf), 'wb'))

                for conf in space_textset_d2v:
                    pickle.dump(model_doc2vec(text, conf), open(text_model_filename(ts.textset_id, 'd2v', conf), 'wb'))

                # update status to completed
                set_key_store('textset:%s:status' % ts.textset_id, 'completed')

