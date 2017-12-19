import logging
from automlk.grapher import launch_grapher
from automlk.context import get_data_folder

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/grapher.log'),
                        logging.StreamHandler()
                    ]
                    )

launch_grapher()
