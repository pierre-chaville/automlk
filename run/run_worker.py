import sys
import logging
from automlk.worker import worker_loop
from automlk.context import get_data_folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)5s [%(module)s %(lineno)3d] %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/worker.log'),
                        logging.StreamHandler()
                    ])

logging.info('starting worker')

if len(sys.argv) > 1:
    wid = sys.argv[1]
else:
    wid = 1

if len(sys.argv) > 2:
    gpu = sys.argv[2]
else:
    gpu = False

worker_loop(wid, gpu)
