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
        return rds.set(str(key), json.dumps(value))
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


def rpush_key_store(key, value):
    # add value to the list of key in store
    if use_redis:
        return rds.rpush(str(key), json.dumps(value))
    else:
        if exists_key_store(key):
            l = get_key_store(key)
            set_key_store(key, l.append(value))
            return l
        else:
            set_key_store(key, [value])


def list_key_store(key):
    # returns the complete list of values
    if use_redis:
        return [json.loads(x) for x in rds.lrange(key, 0, -1)]
    else:
        return get_key_store(key)

