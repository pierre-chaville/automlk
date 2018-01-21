import psutil
import socket
import datetime
from . import __version__
from .store import *

__cpu_pct = 0


def get_heart_beeps(module):
    """
    returns last heart beeps of the controller and workers

    :param module: controller / worker
    :return: list of status or empty list (eg. if environment not set)
    """
    # returns values of heart beeps
    try:
        # get list and values
        l = smembers_key_store('monitor:%s' % module)
        l_hb = [get_key_store('monitor:%s:%s' % (module, id)) for id in l]

        # filter on heart beeps < 12h
        return [h for h in l_hb if
                (datetime.datetime.now() - datetime.datetime(**h['time'])) < datetime.timedelta(hours=12)]
    except:
        return []


def heart_beep(module, msg, index=1, gpu=False):
    """
    send heart beep as module

    :param module: controller / worker
    :param msg: status message to send to the web app
    :param index: index of the worker
    :param gpu: gpu capability
    :return:
    """
    global __cpu_pct
    if module == 'controller':
        __cpu_pct = psutil.cpu_percent(interval=0.1)

    # detect IP and DNS name
    t = datetime.datetime.now()
    id = socket.gethostname() + '_' + str(index)
    msg_beep = {'module': module,
                'host_name': id,
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_log_count': psutil.cpu_count(logical=True),
                'cpu_pct': __cpu_pct,
                'memory': psutil.virtual_memory().total/1073741824,
                'memory_pct': psutil.virtual_memory().percent,
                'version': __version__,
                'time': {'year': t.year, 'month': t.month, 'day': t.day,
                         'hour': t.hour, 'minute': t.minute, 'second': t.second},
                'msg': msg
                }

    # update list
    sadd_key_store('monitor:%s' % module, id)

    # save msg in store
    set_key_store('monitor:%s:%s' % (module, id), msg_beep)
