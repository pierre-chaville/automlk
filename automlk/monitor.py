import psutil
import socket
import datetime
from .store import *


def heart_beep(module, msg):
    # send heart beep as module

    # detect IP and DNS name
    t = datetime.datetime.now()
    id = socket.gethostname()
    msg_beep = {'module': module,
                'host_name': id,
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_log_count': psutil.cpu_count(logical=True),
                'cpu_pct': psutil.cpu_percent(interval=0.1),
                'memory': psutil.virtual_memory().total/1073741824,
                'memory_pct': psutil.virtual_memory().percent,
                'time': {'year': t.year, 'month': t.month, 'day': t.day,
                         'hour': t.hour, 'minute': t.minute, 'second': t.second},
                'msg': msg
                }

    # update list
    sadd_key_store('monitor:%s' % module, id)

    # save msg in store
    set_key_store('monitor:%s:%s' % (module, id), msg_beep)
    print(msg_beep)


def get_heart_beeps(module):
    # returns values of heart beeps

    # get list
    l = smembers_key_store('monitor:%s' % module)

    # get values
    return [get_key_store('monitor:%s:%s' % (module, id)) for id in l]