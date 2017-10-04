import pandas as pd
from automlk.dataset import create_dataset

"""
loads a dataset from a csv file description
"""

for line in pd.read_csv('../datasets/datasets.csv').fillna('').to_dict(orient='records'):
    print('-'*60)
    print('creating dataset %s in %s' % (line['name'], line['problem_type']))
    line['other_metrics'] = line['other_metrics'].replace(' ', '').split(',')
    dt = create_dataset(**line)
