from .context import *
import json

# try to import redis
try:
    import redis
    # force_file_storage = 1/0
    use_redis = True
    rds = redis.Redis(get_config()['store'])
    print('using redis')
except:
    # we will use simple file storage
    use_redis = False
    store_folder = get_data_folder() + '/store'
    print('redis not installed: using file storage instead')


def get_key_store(key):
    # retieves value from key in store
    if use_redis:
        return json.loads(rds.get(str(key)))
    else:
        if exists_key_store(key):
            return json.load(open(store_folder + '/' + key + '.json', 'r'))
        else:
            return None


def set_key_store(key, value):
    # sets value to key in store
    if use_redis:
        rds.set(str(key), json.dumps(value))
    else:
        with open(store_folder + '/' + str(key) + '.json', 'w') as f:
            json.dump(value, f)


def exists_key_store(key):
    # checks if a key exists
    if use_redis:
        return rds.exists(str(key))
    else:
        return os.path.exists(store_folder + '/' + key + '.json')


def incr_key_store(key, amount=1):
    # increments value of key in store with amount
    if use_redis:
        return rds.incr(str(key), amount)
    else:
        if exists_key_store(key):
            value = get_key_store(key)
            set_key_store(key, value + amount)
            return value + amount
        else:
            set_key_store(key, amount)


def get_counter_store(key, amount=1):
    # returns raw value of key in store with amount
    if use_redis:
        if rds.exists(str(key)):
            return int(rds.get(str(key)))
        else:
            return 0
    else:
        if exists_key_store(key):
            return get_key_store(key)
        else:
            return None


def rpush_key_store(key, value):
    # add value to the end of a list of key in store
    if use_redis:
        rds.rpush(str(key), json.dumps(value))
    else:
        if exists_key_store(key):
            l = get_key_store(key)
            set_key_store(key, l.append(value))
        else:
            set_key_store(key, [value])


def rpop_key_store(key):
    # returns and pop the 1st element of a list of key in store
    if use_redis:
        return json.loads(rds.rpop(str(key)))
    else:
        if exists_key_store(key):
            l = get_key_store(key)
            e = l[0]
            set_key_store(key, l[1:])
            return e
        else:
            return None


def brpop_key_store(key):
    # returns and pop the 1st element of a list of key in store with blocking
    if use_redis:
        return json.loads(rds.brpop(str(key))[1])
    else:
        if exists_key_store(key):
            l = get_key_store(key)
            e = l[0]
            set_key_store(key, l[1:])
            return e
        else:
            return None


def lpush_key_store(key, value):
    # add value to the beginning of a list of key in store
    if use_redis:
        rds.lpush(str(key), json.dumps(value))
    else:
        if exists_key_store(key):
            l = get_key_store(key)
            set_key_store(key, [value] + l)
        else:
            set_key_store(key, [value])


def list_key_store(key):
    # returns the complete list of values
    if use_redis:
        return [json.loads(x) for x in rds.lrange(key, 0, -1)]
    else:
        return get_key_store(key)


def llen_key_store(key):
    # returns the length of a list of key in store
    if use_redis:
        return rds.llen(str(key))
    else:
        if exists_key_store(key):
            return len(get_key_store(key))
        else:
            return None
