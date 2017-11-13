from automlk.dataset import *
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from collections import Counter

print('loading dataset')
dt = get_dataset('47')
ds = pickle.load(open(get_dataset_folder(dt.dataset_id) + '/data/eval_set.pkl', 'rb'))

print('resampling RandomOver')
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling regular SMOTE')
ros = SMOTE()
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling RandomUnder')
ros = RandomUnderSampler()
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling ClusterCentroids')
ros = ClusterCentroids()
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))
"""
print('resampling borderline1 SMOTE')
ros = SMOTE(kind='borderline1')
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling borderline2 SMOTE')
ros = SMOTE(kind='borderline2')
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling SVM SMOTE')
ros = SMOTE(kind='svm')
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
print(sorted(Counter(y_resampled).items()))

print('resampling ADASYN')
ros = ADASYN()
X_resampled, y_resampled = ros.fit_sample(ds.X.fillna(0), ds.y)

print(sorted(Counter(ds.y).items()))
"""