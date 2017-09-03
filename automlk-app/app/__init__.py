from flask import Flask, session
import math


def print_list(l):
    # displays a list without double quotes and brackets
    try:
        return ', '.join([str(x) for x in l])
    except:
        return ''


def print_score(x):
    # display score with 5 digits max
    try:
        dim = int(math.log(abs(x), 10))
        if dim > 5:
            return '%.1E' % abs(x)
        else:
            digits = 5-dim
            format = '%6.' + str(digits) + 'f'
            return (format % x).rstrip('0').rstrip('.')
    except:
        return 'N/A'


def print_value(x):
    # easy print function for dictionary value
    return ('%6.4f' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)


def print_duration(y):
    # easy print of a duration in h mn s
    x = float(y)
    if x >= 3600:
        return '%dh%d' % (x // 3600, (x % 3600) // 60)
    elif x >= 60:
        return '%dmn%d' % (x // 60, x % 60)
    else:
        return '%.1fs' % x


def print_params(p):
    # easy print function for dictionary of params
    excluded = ['verbose', 'task', 'n_jobs', 'random_state', 'silent', 'warm_start']
    return ", ".join([key.replace('_', ' ') + ': ' + print_value(p[key]) for key in p.keys() if key not in excluded])


app = Flask(__name__)
SESSION_TYPE = 'redis'
app.config.from_object('config')
app.jinja_env.globals.update(print_list=print_list)
app.jinja_env.globals.update(print_score=print_score)
app.jinja_env.globals.update(print_value=print_value)
app.jinja_env.globals.update(print_duration=print_duration)
app.jinja_env.globals.update(print_params=print_params)

from app import views
