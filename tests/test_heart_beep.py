import time
from automlk.monitor import heart_beep

i = 0
while True:
    i += 1
    heart_beep('worker', {'k1': 1, 'k': i})
    time.sleep(1)