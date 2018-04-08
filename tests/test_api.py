import json
import pandas as pd
import pickle
from automlk.api import *
from automlk.dataset import get_dataset_folder

# parameters
folder = '../../test/'
dataset_id = '7'
round_id = '3'

# load data
j_dataset = json.load(open(folder + 'dataset_%s.json' % dataset_id, 'r'))
j_model = json.load(open(folder + 'model_%s_%s.json' % (dataset_id, round_id), 'r'))
df = pd.read_csv(folder + 'train_%s.csv' % dataset_id)

# fit the model
model = fit_pipeline(j_dataset, j_model, df)
pred = predict_pipeline(j_dataset, model, df)