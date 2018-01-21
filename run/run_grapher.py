import logging
from automlk.grapher import grapher_loop
from automlk.context import get_data_folder
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s %(lineno)d %(message)s',
                    handlers=[
                        logging.FileHandler(get_data_folder() + '/grapher.log'),
                        logging.StreamHandler()
                    ]
                    )
while True:
    try:
        grapher_loop()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error('%s in %s line:%s error: %s' % (exc_type.__name__, fname, str(exc_tb.tb_lineno), str(e)))