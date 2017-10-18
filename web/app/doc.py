import os
import pandas as pd
import numpy as np
from automlk.worker import get_importance
from automlk.context import get_dataset_folder
# from .helper import *

def gener_doc(dataset):
    """
    generate the documentation of this dataset

    :param dataset: dataset object
    :return:
    """
    # check or create doc folder
    folder = get_dataset_folder(dataset.dataset_id) + '/doc'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # then create doc in rst file
    with open(folder + '/index.rst', 'w') as f:
        d = dataset.__dict__
        f.write('Dataset info\n')
        f.write('============\n')
        f.write('\n')
        for k in d.keys():
            if k != 'features':
                f.write('**'+k+':**'+'\n')
                f.write('    *' + str(d[k]) + '*'+'\n')

        # generate features
        f.write('\n')
        f.write('Features\n')
        f.write('--------\n')
        f.write('\n')
        f.write('..include:: features.csv\n')

        df = pd.DataFrame([f.__dict__ for f in dataset.features])
        df[['name', 'description', 'to_keep', 'raw_type', 'col_type', 'n_unique_values', 'n_missing', 'first_unique_values']].to_csv(folder + '/features.csv', index=False)