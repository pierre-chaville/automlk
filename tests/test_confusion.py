import pickle
import pandas as pd
import numpy as np
from automlk.models import get_pred_eval_test
from automlk.dataset import get_dataset_folder
from sklearn.metrics import confusion_matrix

ds = pickle.load(open(get_dataset_folder('4/data/eval_set.pkl'), 'rb'))
y_pred_eval, y_pred_test, y_pred_submit = get_pred_eval_test('4', 15)

m = confusion_matrix(ds.y_train, np.argmax(y_pred_eval, axis=1))

print(m)