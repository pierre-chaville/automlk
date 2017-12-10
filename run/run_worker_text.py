import logging
from automlk.worker_text import launch_worker_text
from automlk.context import get_data_folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(module)s %(lineno)3d] %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/worker.log'),
                        logging.StreamHandler()
                    ])

logging.info('starting worker')

launch_worker_text()
