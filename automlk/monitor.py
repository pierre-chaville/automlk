import psutil
import socket
import time
import _thread
import signal
import datetime
from .store import *
from threading import Timer

__worker_timer_active = False
__worker_timer_start = None
__worker_timer_limit = 0


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


def __timer_control():
    # execute control of max delay
    global __worker_timer_active
    global __worker_timer_start
    global __worker_timer_limit
    if __worker_timer_active:
        # check delay
        t = time.time()
        if t - __worker_timer_start > __worker_timer_limit:
            print('max delay %d seconds reached...' % __worker_timer_limit)
            os.kill(os.getpid(), signal.SIGINT)
            #_thread.interrupt_main()
    Timer(10.0, __timer_control, []).start()


def init_timer_worker():
    # set the timer for the worker to monitor max delays, ...
    global __worker_timer_active
    __worker_timer_active = False
    Timer(10.0, __timer_control, []).start()


def start_timer_worker(limit):
    # set the timer for the worker to monitor max delays, ...
    global __worker_timer_active
    global __worker_timer_start
    global __worker_timer_limit
    __worker_timer_start = time.time()
    __worker_timer_active = True
    __worker_timer_limit = limit


def stop_timer_worker():
    # set the timer for the worker to monitor max delays, ...
    global __worker_timer_active
    __worker_timer_active = False

