import logging
from automlk.controller import launch_controller


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s')

logging.info('starting controller')
launch_controller()

