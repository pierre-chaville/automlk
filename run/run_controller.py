import logging
from automlk.controller import launch_controller
from automlk.context import get_data_folder


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/controller.log'),
                        logging.StreamHandler()
                    ]
                    )

logging.info('starting controller')
launch_controller()

