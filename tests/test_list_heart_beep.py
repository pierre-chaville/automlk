import time
from automlk.monitor import get_heart_beeps

while True:
    print('-'*60)
    for h in get_heart_beeps('worker'):
        print(h)
    time.sleep(1)