from .models import *
import sklearn.ensemble as ske
import sklearn.linear_model as linear
import sklearn.svm as svm
import sklearn.neighbors as knn


class ModelSolution(object):
    # to define a model and the parameters / conditions of usage

    def __init__(self, ref, name, model, default_params, space_params, problem_type, is_wrapper=False,
                 early_stopping=False, use_predict_proba=False, level=1, selectable=True, limit_size=1e32):
        self.ref = ref
        self.name = name
        self.model = model
        self.default_params = default_params
        self.space_params = space_params
        self.problem_type = problem_type
        self.early_stopping = early_stopping
        self.is_wrapper = is_wrapper
        self.use_predict_proba = use_predict_proba
        self.level = level
        self.selectable = selectable
        self.limit_size = limit_size


# list of solutions
model_solutions = [
    # classifiers
    ModelSolution('LGBM-C', 'LightGBM', ModelLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', is_wrapper=True, early_stopping=True,
                  selectable=import_lgbm),
    ModelSolution('XGB-C', 'XgBoost', ModelXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', is_wrapper=True, early_stopping=True,
                  selectable=import_xgb),
    ModelSolution('CAT-C', 'CatBoost', ModelCatboost, default_catboost_classifier,
                  space_catboost_classifier, 'classification', is_wrapper=True, early_stopping=True,
                  selectable=import_catboost),
    ModelSolution('XTRA-C', 'Extra Trees', ske.ExtraTreesClassifier, default_extra_trees, space_extra_trees_classifier,
                  'classification', use_predict_proba=True),
    ModelSolution('RF-C', 'Random Forest', ske.RandomForestClassifier, default_random_forest,
                  space_random_forest_classifier, 'classification', use_predict_proba=True),
    ModelSolution('GBM-C', 'Gradient Boosting', ske.GradientBoostingClassifier, default_gradient_boosting,
                  space_extra_trees_classifier, 'classification', use_predict_proba=True),
    ModelSolution('ADA-C', 'AdaBoost', ske.AdaBoostClassifier, default_adaboost,
                  space_adaboost_classifier, 'classification', use_predict_proba=True),
    ModelSolution('KNN-C', 'Knn', knn.KNeighborsClassifier, default_knn,
                  space_knn, 'classification', use_predict_proba=True, limit_size=10),
    ModelSolution('SVC', 'SVM', svm.SVC, default_svc,
                  space_svc, 'classification', use_predict_proba=True, limit_size=2),
    ModelSolution('LOGIT', 'Logistic Regression', linear.LogisticRegression, default_logistic_regression,
                  space_logistic_regression, 'classification', use_predict_proba=True),
    ModelSolution('NN-C', 'Neural Networks', ModelNN, default_keras,
                  space_keras, 'classification', is_wrapper=True, early_stopping=True, selectable=import_keras,
                  limit_size=10),

    # regressors
    ModelSolution('LGBM-R', 'LightGBM', ModelLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', is_wrapper=True, early_stopping=True, selectable=import_lgbm),
    ModelSolution('XGB-R', 'XgBoost', ModelXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', is_wrapper=True, early_stopping=True, selectable=import_xgb),
    ModelSolution('CAT-R', 'CatBoost', ModelCatboost, default_catboost_regressor,
                  space_catboost_regressor, 'regression', is_wrapper=True, early_stopping=True, selectable=import_catboost),
    ModelSolution('XTRA-R', 'Extra Trees', ske.ExtraTreesRegressor, default_extra_trees,
                  space_extra_trees_regressor, 'regression'),
    ModelSolution('RF-R', 'Random Forest', ske.RandomForestRegressor, default_random_forest,
                  space_random_forest_regressor, 'regression'),
    ModelSolution('GBM-R', 'Gradient Boosting', ske.GradientBoostingRegressor, default_gradient_boosting,
                  space_gradient_boosting_regressor, 'regression'),
    ModelSolution('ADA-R', 'AdaBoost', ske.AdaBoostRegressor, default_adaboost,
                  space_adaboost_regressor, 'regression'),
    ModelSolution('KNN-R', 'Knn', knn.KNeighborsRegressor, default_knn,
                  space_knn, 'regression', limit_size=10),
    ModelSolution('SVR', 'SVM', svm.SVR, default_svr,
                  space_svr, 'regression', limit_size=2),
    ModelSolution('LSVR', 'Linear SVR', svm.LinearSVR, default_linear_svr,
                  space_linear_svr, 'regression', limit_size=10),
    ModelSolution('LR', 'Linear Regression', linear.LinearRegression, default_linear_regression,
                  space_linear_regression, 'regression'),
    ModelSolution('RIDGE', 'Ridge Regression', linear.Ridge, default_ridge_regression,
                  space_ridge_regression, 'regression'),
    ModelSolution('LASSO', 'Lasso Regression', linear.Lasso, default_lasso_regression,
                  space_lasso_regression, 'regression'),
    ModelSolution('HUBER', 'Huber Regression', linear.HuberRegressor, default_huber_regression,
                  space_huber_regression, 'regression'),
    ModelSolution('NN-R', 'Neural Networks', ModelNN, default_keras,
                  space_keras, 'regression', is_wrapper=True, early_stopping=True, selectable=import_keras, limit_size=10),

    # ensembles
    # TODO: fix and improve this model
    # ModelSolution('ENS', 'Ensemble Selection', ModelEnsembleSelection, default_ensemble, space_ensemble, '*', level=2),

    # ensemble classifiers
    ModelSolution('STK-LGBM-C', 'Stacking LightGBM', ModelStackingLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', is_wrapper=True, early_stopping=True, level=2, selectable=import_lgbm),
    ModelSolution('STK-XGB-C', 'Stacking XgBoost', ModelStackingXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', is_wrapper=True, early_stopping=True, level=2, selectable=import_xgb),
    ModelSolution('STK-XTRA-C', 'Stacking Extra Trees', ModelStackingExtraClassifier, default_extra_trees,
                  space_extra_trees_classifier, 'classification', is_wrapper=True, level=2),
    ModelSolution('STK-RF-C', 'Stacking Random Forest', ModelStackingRFClassifier, default_random_forest,
                  space_random_forest_classifier, 'classification', is_wrapper=True, level=2),
    ModelSolution('STK-GBM-C', 'Stacking Gradient Boosting', ModelStackingGBMClassifier, default_gradient_boosting,
                  space_gradient_boosting_classifier, 'classification', is_wrapper=True, level=2),
    ModelSolution('STK-LOGIT', 'Stacking Logistic Regression', ModelStackingLogistic,
                  default_logistic_regression, space_logistic_regression, 'classification', is_wrapper=True, level=2),

    # ensemble regressors
    ModelSolution('STK-LGBM-R', 'Stacking LightGBM', ModelStackingLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', is_wrapper=True, early_stopping=True, level=2, selectable=import_lgbm),
    ModelSolution('STK-XGB-R', 'Stacking XgBoost', ModelStackingXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', is_wrapper=True, early_stopping=True, level=2, selectable=import_xgb),
    ModelSolution('STK-XTRA-R', 'Stacking Extra Trees', ModelStackingExtraRegressor, default_extra_trees,
                  space_extra_trees_regressor, 'regression', is_wrapper=True, level=2),
    ModelSolution('STK-RF-R', 'Stacking Random Forest', ModelStackingRFRegressor, default_random_forest,
                  space_random_forest_regressor, 'regression', is_wrapper=True, level=2),
    ModelSolution('STK-GBM-R', 'Stacking Gradient Boosting', ModelStackingGBMRegressor,
                  default_gradient_boosting, space_gradient_boosting_regressor, 'regression', is_wrapper=True, level=2),
    ModelSolution('STK-LR', 'Stacking Linear Regression', ModelStackingLinear,
                  default_linear_regression, space_linear_regression, 'regression', is_wrapper=True, level=2),
]

# mapping table
model_solutions_map = {s.ref: s for s in model_solutions}
