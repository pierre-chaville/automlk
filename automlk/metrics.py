import math
import sklearn.metrics

# additional metrics not included in sklearn

def rmse(y_act, y_pred):
    return math.sqrt(sklearn.metrics.mean_squared_error(y_act, y_pred))


def rmsle(y_act, y_pred):
    return math.sqrt(sklearn.metrics.mean_squared_log_error(y_act, y_pred))


# metric map as tuple = (function, problem_type, best_is_min)
metric_map = {'log_loss': (sklearn.metrics.log_loss, 'classification', True),
              'accuracy': (sklearn.metrics.accuracy_score, 'classification', False),
              'precision': (sklearn.metrics.precision_score, 'classification', False),
              'recall': (sklearn.metrics.recall_score, 'classification', False),
              'f1': (sklearn.metrics.f1_score, 'classification', False),
              'auc': (sklearn.metrics.auc, 'classification', False),
              'hinge': (sklearn.metrics.hinge_loss, 'classification', True),

              # regression metrics
              'mse': (sklearn.metrics.mean_squared_error, 'regression', True),
              'rmse': (rmse, 'regression', True),
              'mae': (sklearn.metrics.mean_absolute_error, 'regression', True),
              'median': (sklearn.metrics.median_absolute_error, 'regression', True),
              'msle': (sklearn.metrics.mean_squared_log_error, 'regression', True),
              'rmsle': (rmsle, 'regression', True),
              'r2': (sklearn.metrics.r2_score, 'regression', False),
              }
