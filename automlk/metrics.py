import math
import sklearn.metrics


class Metric(object):
    # metric class
    def __init__(self, name, function, problem_type, best_is_min, need_class=False):
        self.name = name
        self.function = function
        self.problem_type = problem_type
        self.best_is_min = best_is_min
        self.need_class = need_class

# additional metrics not included in sklearn


def rmse(y_act, y_pred):
    return math.sqrt(sklearn.metrics.mean_squared_error(y_act, y_pred))


# def rmsle(y_act, y_pred):
#     return math.sqrt(sklearn.metrics.mean_squared_log_error(y_act, y_pred))


# metrics
metric_list = [

    # classification metrics:
    Metric('log_loss', sklearn.metrics.log_loss, 'classification', True),
    Metric('accuracy', sklearn.metrics.accuracy_score, 'classification', False, True),
    Metric('precision', sklearn.metrics.precision_score, 'classification', False, True),
    Metric('recall', sklearn.metrics.recall_score, 'classification', False, True),
    Metric('f1', sklearn.metrics.f1_score, 'classification', False, True),
    Metric('auc', sklearn.metrics.auc, 'classification', False, True),
    Metric('hinge', sklearn.metrics.hinge_loss, 'classification', True),

    # regression metrics
    Metric('mse', sklearn.metrics.mean_squared_error, 'regression', True),
    Metric('rmse', rmse, 'regression', True),
    Metric('mae', sklearn.metrics.mean_absolute_error, 'regression', True),
    Metric('median', sklearn.metrics.median_absolute_error, 'regression', True),
    # Metric('msle', sklearn.metrics.mean_squared_log_error, 'regression', True),
    # Metric('rmsle', rmsle, 'regression', True),
    Metric('r2', sklearn.metrics.r2_score, 'regression', False)
]

metric_map = {m.name: m for m in metric_list}

