Models level 1
--------------

regression:
___________
**LightGBM**
    *task, boosting, learning_rate, num_leaves, tree_learner, max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, drop_rate, skip_drop, max_drop, uniform_drop, xgboost_dart_mode, top_rate, other_rate, verbose, objective, metric*

**XgBoost**
    *booster, eval_metric, eta, min_child_weight, max_depth, gamma, max_delta_step, sub_sample, colsample_bytree, colsample_byleval, lambda, alpha, tree_method, sketch_eps, scale_pos_weight, silent, objective*

**CatBoost**
    *learning_rate, depth, verbose*

**Neural Networks**
    *units, batch_size, batch_normalization, activation, optimizer, learning_rate, number_layers, dropout*

**Extra Trees**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, criterion*

**Random Forest**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion*

**Gradient Boosting**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, learning_rate, loss*

**AdaBoost**
    *n_estimators, learning_rate, random_state, loss*

**Knn**
    *n_neighbors, weights, algorithm, leaf_size, p, n_jobs*

**SVM**
    *C, epsilon, kernel, degree, gamma, coef0, shrinking, tol, max_iter, verbose*

**Linear SVR**
    *C, loss, epsilon, dual, tol, fit_intercept, intercept_scaling, max_iter, verbose*

**Linear Regression**
    *fit_intercept, normalize, copy_X, n_jobs*

**Ridge Regression**
    *alpha, fit_intercept, normalize, copy_X, tol, solver*

**Lasso Regression**
    *alpha, fit_intercept, normalize, precompute, copy_X, tol, positive, selection*

**Huber Regression**
    *epsilon, alpha, fit_intercept, tol*


classification:
_______________
**LightGBM**
    *task, boosting, learning_rate, num_leaves, tree_learner, max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, drop_rate, skip_drop, max_drop, uniform_drop, xgboost_dart_mode, top_rate, other_rate, verbose, objective, metric*

**XgBoost**
    *booster, eval_metric, eta, min_child_weight, max_depth, gamma, max_delta_step, sub_sample, colsample_bytree, colsample_byleval, lambda, alpha, tree_method, sketch_eps, scale_pos_weight, silent, objective*

**CatBoost**
    *learning_rate, depth, verbose*

**Extra Trees**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion, class_weight*

**Random Forest**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion, class_weight*

**Gradient Boosting**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, learning_rate, criterion, loss*

**AdaBoost**
    *n_estimators, learning_rate, random_state, algorithm*

**Knn**
    *n_neighbors, weights, algorithm, leaf_size, p, n_jobs*

**SVM**
    *C, kernel, degree, gamma, coef0, shrinking, tol, max_iter, verbose, probability*

**Logistic Regression**
    *penalty, dual, tol, C, fit_intercept, intercept_scaling, solver, max_iter, multi_class, n_jobs*

**NB Gaussian**
    **

**NB Bernoulli**
    *alpha, binarize, fit_prior*

**Neural Networks**
    *units, batch_size, batch_normalization, activation, optimizer, learning_rate, number_layers, dropout*


Ensembles
---------

regression:
___________
**Stacking LightGBM**
    *task, boosting, learning_rate, num_leaves, tree_learner, max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, drop_rate, skip_drop, max_drop, uniform_drop, xgboost_dart_mode, top_rate, other_rate, verbose, objective, metric*

**Stacking XgBoost**
    *booster, eval_metric, eta, min_child_weight, max_depth, gamma, max_delta_step, sub_sample, colsample_bytree, colsample_byleval, lambda, alpha, tree_method, sketch_eps, scale_pos_weight, silent, objective*

**Stacking Extra Trees**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, criterion*

**Stacking Random Forest**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion*

**Stacking Gradient Boosting**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, learning_rate, loss*

**Stacking Linear Regression**
    *fit_intercept, normalize, copy_X, n_jobs*


classification:
_______________
**Stacking LightGBM**
    *task, boosting, learning_rate, num_leaves, tree_learner, max_depth, min_data_in_leaf, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, bagging_freq, lambda_l1, lambda_l2, min_gain_to_split, drop_rate, skip_drop, max_drop, uniform_drop, xgboost_dart_mode, top_rate, other_rate, verbose, objective, metric*

**Stacking XgBoost**
    *booster, eval_metric, eta, min_child_weight, max_depth, gamma, max_delta_step, sub_sample, colsample_bytree, colsample_byleval, lambda, alpha, tree_method, sketch_eps, scale_pos_weight, silent, objective*

**Stacking Neural Networks**
    *units, batch_size, batch_normalization, activation, optimizer, learning_rate, number_layers, dropout*

**Stacking Extra Trees**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion, class_weight*

**Stacking Random Forest**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, n_jobs, criterion, class_weight*

**Stacking Gradient Boosting**
    *n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, verbose, random_state, warm_start, learning_rate, criterion, loss*

**Stacking Logistic Regression**
    *penalty, dual, tol, C, fit_intercept, intercept_scaling, solver, max_iter, multi_class, n_jobs*

**Stacking Neural Networks**
    *units, batch_size, batch_normalization, activation, optimizer, learning_rate, number_layers, dropout*

