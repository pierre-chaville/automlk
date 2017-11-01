import math
import numpy as np
import sklearn.metrics


class Metric(object):
    # metric class
    def __init__(self, name, function, problem_type, best_is_min, need_class=False, binary=False):
        self.name = name
        self.function = function
        self.problem_type = problem_type
        self.best_is_min = best_is_min
        self.need_class = need_class
        self.binary = binary

# additional metrics not included in sklearn


def rmse(y_act, y_pred):
    """
    metrics rmse = Root Mean Squared Error (regression only)

    :param y_act: vector of actual values
    :param y_pred: vector of predicted values
    :return: rmse
    """
    return math.sqrt(sklearn.metrics.mean_squared_error(y_act, y_pred))


def rmsle(y_act, y_pred):
    """
    metrics rmsle = Root Mean Squared Log Error (regression only)

    :param y_act: vector of actual values
    :param y_pred: vector of predicted values
    :return: rmsle
    """
    return math.sqrt(sklearn.metrics.mean_squared_log_error(y_act, y_pred))


def gini(y_act, y_pred):
    """
    metrics gini = Gini coefficient (classification only)

    :param y_act: vector of actual values
    :param y_pred: vector of predicted values
    :return: gini
    """
    assert (len(y_act) == len(y_pred))
    all = np.asarray(np.c_[y_act, y_pred, np.arange(len(y_act))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(y_act) + 1) / 2.
    return giniSum / len(y_act)


def gini_normalized(y_act, y_pred):
    """
    metrics normalized gini = Normalized Gini coefficient (classification only)

    :param y_act: vector of actual values
    :param y_pred: vector of predicted values
    :return: gini
    """
    return gini(y_act, y_pred) / gini(y_act, y_act)


# metrics
metric_list = [

    # classification metrics:
    Metric('log_loss', sklearn.metrics.log_loss, 'classification', True),
    Metric('accuracy', sklearn.metrics.accuracy_score, 'classification', False, need_class=True),
    Metric('precision', sklearn.metrics.precision_score, 'classification', False, need_class=True),
    Metric('recall', sklearn.metrics.recall_score, 'classification', False, need_class=True),
    Metric('f1', sklearn.metrics.f1_score, 'classification', False, need_class=True),
    Metric('auc', sklearn.metrics.roc_auc_score, 'classification', False, need_class=False, binary=True),
    Metric('hinge', sklearn.metrics.hinge_loss, 'classification', True),
    Metric('gini', gini, 'classification', False, need_class=True, binary=True),
    Metric('gini_norm', gini_normalized, 'classification', False, need_class=False, binary=True),

    # regression metrics
    Metric('mse', sklearn.metrics.mean_squared_error, 'regression', True),
    Metric('rmse', rmse, 'regression', True),
    Metric('mae', sklearn.metrics.mean_absolute_error, 'regression', True),
    Metric('median', sklearn.metrics.median_absolute_error, 'regression', True),
    Metric('msle', sklearn.metrics.mean_squared_log_error, 'regression', True),
    Metric('rmsle', rmsle, 'regression', True),
    Metric('r2', sklearn.metrics.r2_score, 'regression', False)
]

metric_map = {m.name: m for m in metric_list}

