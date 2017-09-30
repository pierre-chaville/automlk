import random
import time

from automlk.store import *

while True:
    if llen_key_store('test_queue') == 0:
        # send queue a random int
        msg = 'test_msg_%d' % random.randint(1, 10)
        print('sending %s' % msg)
        lpush_key_store('test_queue', msg)

        # wait random delay
        delay = random.randint(1,10)
        time.sleep(random.randint(1,10))
    else:
        time.sleep(1)
