from .models import *


class ModelSolution(object):
    # to define a model and the parameters / conditions of usage

    def __init__(self, ref, name, model, default_params, space_params, problem_type, level=1, selectable=True, limit_size=1e32):
        self.ref = ref
        self.name = name
        self.model = model
        self.default_params = default_params
        self.space_params = space_params
        self.problem_type = problem_type
        self.level = level
        self.selectable = selectable
        self.limit_size = limit_size


# list of solutions
model_solutions = [
    # classifiers
    ModelSolution('LGBM-C', 'LightGBM', HyperModelLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', selectable=import_lgbm),
    ModelSolution('XGB-C', 'XgBoost', HyperModelXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', selectable=import_xgb),
    ModelSolution('CAT-C', 'CatBoost', HyperModelCatboost, default_catboost_classifier,
                  space_catboost_classifier, 'classification', selectable=import_catboost),
    ModelSolution('NN-C', 'Neural Networks', HyperModelNN, default_keras,
                  space_keras, 'classification', selectable=import_keras),
    ModelSolution('XTRA-C', 'Extra Trees', HyperModelExtraTrees, default_extra_trees,
                  space_extra_trees_classifier, 'classification'),
    ModelSolution('RF-C', 'Random Forest', HyperModelRandomForest, default_random_forest,
                  space_random_forest_classifier, 'classification'),
    ModelSolution('GBM-C', 'Gradient Boosting', HyperModelExtraTrees, default_extra_trees,
                  space_extra_trees_classifier, 'classification'),
    ModelSolution('ADA-C', 'AdaBoost', HyperModelAdaBoost, default_adaboost,
                  space_adaboost_classifier, 'classification'),
    ModelSolution('KNN-C', 'Knn', HyperModelKnn, default_knn,
                  space_knn, 'classification'),
    ModelSolution('SVC', 'SVM', HyperModelSVM, default_svc,
                  space_svc, 'classification'),
    ModelSolution('LOGIT', 'Logistic Regression', HyperModelLogisticRegression, default_logistic_regression,
                  space_logistic_regression, 'classification', limit_size=2000),

    # regressors
    ModelSolution('LGBM-R', 'LightGBM', HyperModelLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', selectable=import_lgbm),
    ModelSolution('XGB-R', 'XgBoost', HyperModelXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', selectable=import_xgb),
    ModelSolution('CAT-R', 'CatBoost', HyperModelCatboost, default_catboost_regressor,
                  space_catboost_regressor, 'regression', selectable=import_catboost),
    ModelSolution('NN-R', 'Neural Networks', HyperModelNN, default_keras,
                  space_keras, 'classification', selectable=import_keras),
    ModelSolution('XTRA-R', 'Extra Trees', HyperModelExtraTrees, default_extra_trees,
                  space_extra_trees_regressor, 'regression'),
    ModelSolution('RF-R', 'Random Forest', HyperModelRandomForest, default_random_forest,
                  space_random_forest_regressor, 'regression'),
    ModelSolution('GBM-R', 'Gradient Boosting', HyperModelExtraTrees, default_extra_trees,
                  space_extra_trees_regressor, 'regression'),
    ModelSolution('ADA-R', 'AdaBoost', HyperModelAdaBoost, default_adaboost,
                  space_adaboost_regressor, 'regression'),
    ModelSolution('KNN-R', 'Knn', HyperModelKnn, default_knn,
                  space_knn, 'regression'),
    ModelSolution('SVR', 'SVM', HyperModelSVM, default_svr,
                  space_svr, 'regression'),
    ModelSolution('LSVR', 'Linear SVR', HyperModelLinearSVR, default_linear_svr,
                  space_linear_svr, 'regression'),
    ModelSolution('LR', 'Linear Regression', HyperModelLinearRegressor, default_linear_regression,
                  space_linear_regression, 'regression'),
    ModelSolution('RIDGE', 'Ridge Regression', HyperModelRidgeRegressor, default_ridge_regression,
                  space_ridge_regression, 'regression'),
    ModelSolution('LASSO', 'Lasso Regression', HyperModelLassoRegressor, default_lasso_regression,
                  space_lasso_regression, 'regression'),
    ModelSolution('HUBER', 'Huber Regression', HyperModelHuberRegressor, default_huber_regression,
                  space_huber_regression, 'regression'),

    # ensembles
    ModelSolution('ENS', 'Ensemble Selection', HyperModelEnsembleSelection, default_ensemble,
                  space_ensemble, '*', level=2),

    # ensemble classifiers
    ModelSolution('STK-LGBM-C', 'Stacking LightGBM', HyperModelStackingLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', level=2, selectable=import_lgbm),
    ModelSolution('STK-XGB-C', 'Stacking XgBoost', HyperModelStackingXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', level=2, selectable=import_xgb),
    ModelSolution('STK-XTRA-C', 'Stacking Extra Trees', HyperModelStackingExtraTrees, default_extra_trees,
                  space_extra_trees_classifier, 'classification', level=2),
    ModelSolution('STK-RF-C', 'Stacking Random Forest', HyperModelStackingRandomForest, default_random_forest,
                  space_random_forest_classifier, 'classification', level=2),
    ModelSolution('STK-GBM-C', 'Stacking Gradient Boosting', HyperModelStackingGradientBoosting, default_gradient_boosting,
                  space_gradient_boosting_classifier, 'classification', level=2),
    ModelSolution('STK-LOGIT', 'Stacking Logistic Regression', HyperModelStackingLogistic,
                  default_logistic_regression, space_logistic_regression, 'classification', level=2),

    # ensemble regressors
    ModelSolution('STK-LGBM-R', 'Stacking LightGBM', HyperModelStackingLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', level=2, selectable=import_lgbm),
    ModelSolution('STK-XGB-R', 'Stacking XgBoost', HyperModelStackingXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', level=2, selectable=import_xgb),
    ModelSolution('STK-XTRA-R', 'Stacking Extra Trees', HyperModelStackingExtraTrees, default_extra_trees,
                  space_extra_trees_regressor, 'regression', level=2),
    ModelSolution('STK-RF-R', 'Stacking Random Forest', HyperModelStackingRandomForest, default_random_forest,
                  space_random_forest_regressor, 'regression', level=2),
    ModelSolution('STK-GBM-R', 'Stacking Gradient Boosting', HyperModelStackingGradientBoosting,
                  default_gradient_boosting, space_gradient_boosting_regressor, 'regression', level=2),
    ModelSolution('STK-LR', 'Stacking Linear Regression', HyperModelStackingLinear,
                  default_linear_regression, space_linear_regression, 'regression', level=2),

]

# mapping table
model_solutions_map = {s.ref: s for s in model_solutions}
