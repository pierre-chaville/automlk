import pandas as pd
from automlk.dataset import DataSet

"""
loads a dataset from a csv file description
"""

for line in pd.read_csv('../datasets/datasets1.csv').fillna('').to_dict(orient='records'):
    print('creating dataset %s in %s' % (line['name'], line['problem_type']))
    dt = DataSet(name=line['name'],
                 description=line['description'],
                 problem_type=line['problem_type'],
                 y_col=line['y_col'],
                 is_uploaded=line['is_uploaded'],
                 source=line['source'],
                 filename_train=line['filename_train'],
                 metric=line['metric'],
                 filename_test=line['filename_test'],
                 is_public=line['is_public'],
                 url=line['url'],
                 val_col=line['val_col'],
                 val_col_shuffle=line['val_col_shuffle'],
                 other_metrics=line['other_metrics'].replace(' ', '').split(','),
                 cv_folds=line['cv_folds'],
                 holdout_ratio=line['holdout_ratio']
                 )
