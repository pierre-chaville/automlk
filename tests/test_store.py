from automlk.store import *

for key, val in [['test1', 12], ['test2', 'hello'], ['test3', True], ['test4', 56.45454], ['test5', {'a':1}]]:
    print(key, val)
    set_key_store(key, val)
    r = get_key_store(key)
    print(r)
    print(type(r))
    print('-'*60)

print('exists dataset:counter ? ', exists_key_store('dataset:counter'))

print(list_key_store('test_list'))

rpush_key_store('test_list', 121)
rpush_key_store('test_list', 341)
rpush_key_store('test_list', 561)

print(list_key_store('test_list'))

