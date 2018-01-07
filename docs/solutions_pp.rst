categorical encoding:
---------------------

**No encoding**
    **

**Label Encoder**
    **

**One hot categorical**
    *drop_invariant*

**BaseN categorical**
    *drop_invariant, base*

**Hashing categorical**
    *drop_invariant*


text encoding:
--------------

**Bag of words**


**Word2Vec**


**Doc2Vec**



imputing missing values:
------------------------

**No missing**
    **

**Missing values fixed**
    *fixed*

**Missing values frequencies**
    *frequency*


feature scaling:
----------------

**No scaling**
    **

**Scaling Standard**
    **

**Scaling MinMax**
    **

**Scaling MaxAbs**
    **

**Scaling Robust**
    *quantile_range*


feature selection:
------------------

**No Feature selection**
    **

**Truncated SVD**
    *n_components, algorithm*

**Fast ICA**
    *n_components, algorithm*

**PCA**
    *n_components*

**Selection RF**
    *n_estimators*

**Selection RF**
    *n_estimators*

**Selection LSVR**
    **


