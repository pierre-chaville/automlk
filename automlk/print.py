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
        dim = max(0, int(math.log(abs(x), 10)))
        if dim > 5:
            return '%.1E' % abs(x)
        else:
            digits = 5-dim
            fmt = '%6.' + str(digits) + 'f'
            return (fmt % abs(x)).rstrip('0').rstrip('.')
    except:
        return 'N/A'


def print_score_std(x):
    # display std score with 4 digits max
    try:
        dim = max(0, int(math.log(abs(x), 10)))
        if dim > 4:
            return '%.1E' % abs(x)
        else:
            digits = 4-dim
            fmt = '%5.' + str(digits) + 'f'
            return (fmt % abs(x)).rstrip('0').rstrip('.')
    except:
        return 'N/A'


def print_other_metrics(x):
    # display other metrics
    try:
        return ", ".join([y+': '+print_score(x[y]) for y in x.keys()])
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
    if p != None:
        return ", ".join([key.replace('_', ' ') + ': ' + print_value(p[key]) for key in p.keys() if key not in excluded])
    else:
        return None

