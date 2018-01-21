"""
rules to check a set of parameters

"""


def rule_logistic(dataset, params):
    """
    check params for logistic regression specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss
    # ‘liblinear’ is limited to one-versus-rest schemes
    if params['solver'] == 'liblinear':
        if dataset.y_n_classes == 2:
            # Solver liblinear does not support a multinomial backend
            params['multi_class'] = 'ovr'
        else:
            params['solver'] = 'newton-cg'

    # Dual formulation is only implemented for l2 penalty with liblinear solver
    if not (params['solver'] == 'liblinear' and params['penalty'] == 'l2'):
        params['dual'] = False

    # ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
    if params['solver'] in ['newton-cg', 'sag', 'lbfgs']:
        params['penalty'] = 'l2'

    return params


def rule_gbm(dataset, params):
    """
    check params for gradient boosting specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if dataset.problem_type == 'classification' and dataset.y_n_classes > 2:
        params['loss'] = 'deviance'
    return params


def rule_linear_svr(dataset, params):
    """
    check params for logistic regression specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if params['loss'] == 'squared_epsilon_insensitive':
        params['dual'] = False

    if params['loss'] == 'epsilon_insensitive':
        params['dual'] = True

    return params


def rule_catboost(dataset, params):
    """
    check params for logistic regression specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if dataset.problem_type == 'classification':
        if dataset.y_n_classes == 2:
            params['loss_function'] = 'Logloss'
        else:
            params['loss_function'] = 'MultiClass'

    return params


def rule_lightgbm(dataset, params):
    """
    check params for lightgbm specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if dataset.problem_type == 'classification' and dataset.y_n_classes > 2:
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['num_class'] = dataset.y_n_classes

    # updates params according to Light GBM rules
    if 'bagging_freq' in params and params['bagging_freq'] == 0:
        params.pop('bagging_freq', None)
    if 'boosting' in params and params['boosting'] == 'goss':
        params.pop('bagging_freq', None)
        params.pop('bagging_fraction', None)

    return params


def rule_xgboost(dataset, params):
    """
    check params for xgboost specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if dataset.problem_type == 'classification' and dataset.y_n_classes > 2:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        params['num_class'] = dataset.y_n_classes

    return params


def rule_nn(dataset, params):
    """
    check params for neural networks specifically

    :param dataset:
    :param params: params
    :return: updated params
    """
    if dataset.problem_type == 'regression':
        params['output_dim'] = 1
    else:
        params['output_dim'] = dataset.y_n_classes

    return params
