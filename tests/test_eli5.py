import pickle
import eli5
import pandas as pd
from eli5.sklearn import PermutationImportance
from automlk.context import get_dataset_folder
from automlk.dataset import get_dataset, get_dataset_sample


dataset_id = '37'
round_id = '19'

dataset = get_dataset(dataset_id)
ds = pickle.load(open(get_dataset_folder(dataset_id) + '/data/eval_set.pkl', 'rb'))

folder = get_dataset_folder(dataset_id) + '/models/'

names = list(pickle.load(open(folder + '%s_feature_names.pkl' % round_id, 'rb')))
print(names)

model = pickle.load(open(folder + '%s_model.pkl' % round_id, 'rb'))
pipe_model = pickle.load(open(folder + '%s_pipe_model.pkl' % round_id, 'rb'))
pipe_transform = pickle.load(open(folder + '%s_pipe_transform.pkl' % round_id, 'rb'))

sample = get_dataset_sample(dataset_id)

X_sample = pipe_transform.transform(pd.DataFrame(sample)[dataset.x_cols])
"""
print('-'*60)
print('test prediction pipeline')
print(pd.DataFrame(sample))

pipe_model.predict(pd.DataFrame(sample)[dataset.x_cols])
"""
print('-'*60)

print('explain model only')
exp = eli5.explain_weights(model, feature_names=names)
with open('eli5_model.html', 'w') as f:
    f.write(eli5.format_as_html(exp))

print(eli5.format_as_text(exp))

print('-'*60)
print('full pipeline')
print(pipe_model.steps)

exp = eli5.explain_weights(pipe_model)
with open('eli5_pipe_model.html', 'w') as f:
    f.write(eli5.format_as_html(exp))

print(eli5.format_as_text(exp))

print('-'*60)
print('explain black box')
perm = PermutationImportance(model, scoring='r2').fit(pipe_transform.transform(ds.X_test), ds.y_test)
exp = eli5.explain_weights(perm, feature_names=names)

print(eli5.format_as_text(exp))
"""
print('-'*60)
print('explain black box (full pipeline)')
perm = PermutationImportance(pipe_model, scoring='r2').fit(ds.X_test, ds.y_test)
exp = eli5.explain_weights(perm)

print(eli5.format_as_text(exp))
"""

print('-'*60)
print('explain prediction (full pipeline)')

exp = eli5.explain_prediction(pipe_model, sample[0])
with open('eli5_explain_full.html', 'w') as f:
    f.write(eli5.format_as_html(exp))

print(eli5.format_as_text(exp))

print('-'*60)
print('explain prediction (model)')

exp = eli5.explain_prediction(model, X_sample.iloc[0])
with open('eli5_explain.html', 'w') as f:
    f.write(eli5.format_as_html(exp))

print(eli5.format_as_text(exp))
