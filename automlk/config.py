METRIC_NULL = 1e7
SEARCH_QUEUE = 'controller:search_queue'
DUPLICATE_QUEUE = 'controller:duplicate_queue'
RESULTS_QUEUE = 'controller:results_queue'
CONTROLLER_ID = 'controller:dataset_id'

USE_REDIS = False


def set_use_redis(value):
    """
    set global value USE_REDIS

    :param value:
    :return:
    """
    global USE_REDIS
    USE_REDIS = value


def get_use_redis():
    """
    get global value USE_REDIS
    :return: value USE_REDIS
    """
    return USE_REDIS