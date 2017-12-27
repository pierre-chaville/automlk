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


def print_rounded(x, r):
    # print floated rounded
    return str((round(x, r))).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)


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
    if isinstance(p, dict):
        return ", ".join([key.replace('_', ' ') + ': ' + print_value(p[key]) for key in p.keys() if key not in excluded])
    else:
        return None


def print_indent(t, indent=4):
    """
    print a text (eg a python piece of code) with indent
    :param t: text (string with \n)
    :param indent: indentation
    :return: formatted text
    """
    s_indent = "&nbsp;"*indent
    return s_indent + t.replace('\n', '<br>' + s_indent).replace('\t', s_indent)


def print_title(t, s):
    # prints a title t in rst format, using the symbole s
    return t + '\n' #+ s*len(s) + '\n'


def print_summary(dataset):
    # print a summary of the dataset
    s = '%dK rows x %d cols' % (dataset.n_rows, dataset.n_cols)
    if dataset.n_cat_cols > 0:
        s += ', %d categ. cols' % (dataset.n_cat_cols)
    if dataset.n_missing > 0:
        s += ', %d missing cols' % (dataset.n_missing)
    if len(dataset.text_cols) > 0:
        s += ', %d text cols' % (len(dataset.text_cols))
    return s