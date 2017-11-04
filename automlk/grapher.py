import time
from .store import *
from .monitor import heart_beep
from .dataset import create_graph_data


def launch_grapher():
    """
    periodically pool the grapher queue for a job

    :return: 
    """
    while True:
        # poll queue
        msg_search = brpop_key_store('grapher:queue')
        heart_beep('grapher', msg_search)
        if msg_search != None:
            print('reveived %s' % msg_search)
            create_graph_data(msg_search['dataset_id'])



