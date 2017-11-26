categorical encoding:
---------------------

**One hot categorical**
    *drop_invariant*

**BaseN categorical**
    *drop_invariant, base*

**Hashing categorical**
    *drop_invariant*


text encoding:
--------------

**Bag of words**
    *max_features, ngram_range, tfidf, first_words*

**Word2Vec**
    *size, iter, window, min_count, sg, workers*

**Doc2Vec**
    *size, iter, window, min_count, dm, workers*


imputing missing values:
------------------------

**Missing values fixed**
    *fixed*

**Missing values**
    *strategy*


feature scaling:
----------------

**Feature Scaling**
    *scaler*

**No Scaling**
    **


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


sampling:
---------

**No re-sampling**
    **

**Random Over**
    **

**SMOTE**
    **

**Random Under**
    **


