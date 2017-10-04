import random
import time

from automlk.store import *

while True:
    # send queue a random int
    msg = brpop_key_store('test_queue')
    if msg != None:
        print('reveived %s' % msg)

        # wait 1s
        time.sleep(1)
