from .models import *
from .spaces.rules import *
import sklearn.ensemble as ske
import sklearn.linear_model as linear
import sklearn.svm as svm
import sklearn.neighbors as knn
import sklearn.naive_bayes as nb
from .solutions_pp import *


try:
    import lightgbm as lgb

    import_lgbm = True
except:
    import_lgbm = False
    log.info('could not import LightGBM. This model will not be used')

try:
    import xgboost as xgb

    import_xgb = True
except:
    import_xgb = False
    log.info('could not import Xgboost. This model will not be used')


class ModelSolution(object):
    # to define a model and the parameters / conditions of usage

    def __init__(self, ref, name, model, default_params, space_params, problem_type, is_wrapper=False,
                 use_early_stopping=False, early_stopping='', use_predict_proba=False, level=1, selectable=True,
                 limit_size=1e32,
                 rule_params=None, pp_default=[], pp_list=[]):
        self.ref = ref
        self.name = name
        self.model = model
        self.default_params = default_params
        self.space_params = space_params
        self.problem_type = problem_type
        self.use_early_stopping = use_early_stopping
        self.early_stopping = early_stopping
        self.is_wrapper = is_wrapper
        self.use_predict_proba = use_predict_proba
        self.level = level
        self.selectable = selectable
        self.limit_size = limit_size
        self.rule_params = rule_params
        self.pp_default = pp_default
        self.pp_list = pp_list

"""
    ModelSolution('LGBM-C', 'LightGBM', ModelLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', is_wrapper=True, use_early_stopping=True,
                  rule_params=rule_lightgbm, selectable=import_lgbm, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('XGB-C', 'XgBoost', ModelXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', is_wrapper=True, use_early_stopping=True,
                  rule_params=rule_xgboost, selectable=import_xgb, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('LGBM-R', 'LightGBM', ModelLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', is_wrapper=True, use_early_stopping=True,
                  rule_params=rule_lightgbm, selectable=import_lgbm, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('XGB-R', 'XgBoost', ModelXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', is_wrapper=True, use_early_stopping=True,
                  rule_params=rule_xgboost, selectable=import_xgb, pp_default=pp_def_trees, pp_list=pp_list_trees),
"""

# list of solutions
model_solutions = [
    # classifiers
    ModelSolution('LGBM-C', 'LightGBM', lgb.LGBMClassifier, default_sk_lightgbm_classifier,
                  space_sk_lightgbm_classifier, 'classification', is_wrapper=False, use_early_stopping=True,
                  early_stopping='LGBM', rule_params=rule_lightgbm, selectable=import_lgbm, pp_default=pp_def_lgbm, pp_list=pp_list_lgbm),
    ModelSolution('XGB-C', 'XgBoost', xgb.XGBClassifier, default_sk_xgboost_classifier,
                  space_sk_xgboost_classifier, 'classification', is_wrapper=False, use_early_stopping=True,
                  early_stopping='XGB', rule_params=rule_xgboost, selectable=import_xgb, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('CAT-C', 'CatBoost', ModelCatboost, default_catboost_classifier,
                  space_catboost_classifier, 'classification', is_wrapper=True, use_early_stopping=True, rule_params=rule_catboost,
                  selectable=import_catboost, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('XTRA-C', 'Extra Trees', ske.ExtraTreesClassifier, default_extra_trees, space_extra_trees_classifier,
                  'classification', use_predict_proba=True, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('RF-C', 'Random Forest', ske.RandomForestClassifier, default_random_forest,
                  space_random_forest_classifier, 'classification', use_predict_proba=True, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('GBM-C', 'Gradient Boosting', ske.GradientBoostingClassifier, default_gradient_boosting,
                  space_gradient_boosting_classifier, 'classification', rule_params=rule_gbm,
                  use_predict_proba=True, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('ADA-C', 'AdaBoost', ske.AdaBoostClassifier, default_adaboost,
                  space_adaboost_classifier, 'classification', use_predict_proba=True, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('KNN-C', 'Knn', knn.KNeighborsClassifier, default_knn,
                  space_knn, 'classification', use_predict_proba=True, limit_size=10, pp_default=pp_def_knn, pp_list=pp_list_knn),
    ModelSolution('SVC', 'SVM', svm.SVC, default_svc,
                  space_svc, 'classification', use_predict_proba=True, limit_size=2, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('LOGIT', 'Logistic Regression', linear.LogisticRegression, default_logistic_regression,
                  space_logistic_regression, 'classification', use_predict_proba=True,
                  rule_params=rule_logistic, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('NB-GAUSS', 'Naive Bayes Gaussian', nb.GaussianNB, {}, {}, 'classification',
                  use_predict_proba=True, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('NB-BERN', 'Naive Bayes  Bernoulli', nb.BernoulliNB, default_nb_bernoulli, space_nb_bernoulli,
                  'classification', use_predict_proba=True, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('NN-C', 'Neural Networks', ModelNN, default_keras,
                  space_keras, 'classification', is_wrapper=True, use_early_stopping=True,  rule_params=rule_nn,
                  selectable=import_keras, limit_size=100, pp_default=pp_def_NN, pp_list=pp_list_NN),

    # regressors
    ModelSolution('LGBM-R', 'LightGBM', lgb.LGBMRegressor, default_sk_lightgbm_regressor,
                  space_sk_lightgbm_regressor, 'regression', is_wrapper=False, use_early_stopping=True,
                  early_stopping='LGBM', rule_params=rule_lightgbm, selectable=import_lgbm, pp_default=pp_def_lgbm, pp_list=pp_list_lgbm),
    ModelSolution('XGB-R', 'XgBoost', xgb.XGBRegressor, default_sk_xgboost_regressor,
                  space_sk_xgboost_regressor, 'regression', is_wrapper=False, use_early_stopping=True,
                  early_stopping='XGB', rule_params=rule_xgboost, selectable=import_xgb, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('CAT-R', 'CatBoost', ModelCatboost, default_catboost_regressor,
                  space_catboost_regressor, 'regression', is_wrapper=True, use_early_stopping=True,
                  rule_params=rule_catboost, selectable=import_catboost, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('NN-R', 'Neural Networks', ModelNN, default_keras,
                  space_keras, 'regression', is_wrapper=True, use_early_stopping=True, rule_params=rule_nn,
                  selectable=import_keras, limit_size=100, pp_default=pp_def_NN, pp_list=pp_list_NN),
    ModelSolution('XTRA-R', 'Extra Trees', ske.ExtraTreesRegressor, default_extra_trees,
                  space_extra_trees_regressor, 'regression', pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('RF-R', 'Random Forest', ske.RandomForestRegressor, default_random_forest,
                  space_random_forest_regressor, 'regression', pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('GBM-R', 'Gradient Boosting', ske.GradientBoostingRegressor, default_gradient_boosting,
                  space_gradient_boosting_regressor, 'regression', rule_params=rule_gbm, pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('ADA-R', 'AdaBoost', ske.AdaBoostRegressor, default_adaboost,
                  space_adaboost_regressor, 'regression', pp_default=pp_def_trees, pp_list=pp_list_trees),
    ModelSolution('KNN-R', 'Knn', knn.KNeighborsRegressor, default_knn,
                  space_knn, 'regression', limit_size=10, pp_default=pp_def_knn, pp_list=pp_list_knn),
    ModelSolution('SVR', 'SVM', svm.SVR, default_svr,
                  space_svr, 'regression', limit_size=2, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('LSVR', 'Linear SVR', svm.LinearSVR, default_linear_svr,
                  space_linear_svr, 'regression', rule_params=rule_linear_svr, limit_size=10, pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('LR', 'Linear Regression', linear.LinearRegression, default_linear_regression,
                  space_linear_regression, 'regression', pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('RIDGE', 'Ridge Regression', linear.Ridge, default_ridge_regression,
                  space_ridge_regression, 'regression', pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('LASSO', 'Lasso Regression', linear.Lasso, default_lasso_regression,
                  space_lasso_regression, 'regression', pp_default=pp_def_linear, pp_list=pp_list_linear),
    ModelSolution('HUBER', 'Huber Regression', linear.HuberRegressor, default_huber_regression,
                  space_huber_regression, 'regression', pp_default=pp_def_linear, pp_list=pp_list_linear),

    # ensembles
    # TODO: fix and improve this model
    # ModelSolution('ENS', 'Ensemble Selection', ModelEnsembleSelection, default_ensemble, space_ensemble, '*', level=2),

    # ensemble classifiers
    ModelSolution('STK-LGBM-C', 'Stacking LightGBM', ModelLightGBM, default_lightgbm_classifier,
                  space_lightgbm_classifier, 'classification', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_lightgbm, selectable=import_lgbm),
    ModelSolution('STK-XGB-C', 'Stacking XgBoost', ModelXgBoost, default_xgboost_classifier,
                  space_xgboost_classifier, 'classification', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_xgboost, selectable=import_xgb),
    ModelSolution('STK-NN-C', 'Stacking Neural Networks', ModelNN, default_keras,
                  space_keras, 'classification', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_nn, selectable=import_keras, limit_size=100),
    ModelSolution('STK-XTRA-C', 'Stacking Extra Trees', ske.ExtraTreesClassifier, default_extra_trees,
                  space_extra_trees_classifier, 'classification', use_predict_proba=True, level=2),
    ModelSolution('STK-RF-C', 'Stacking Random Forest', ske.RandomForestClassifier, default_random_forest,
                  space_random_forest_classifier, 'classification', use_predict_proba=True, level=2),
    ModelSolution('STK-GBM-C', 'Stacking Gradient Boosting', ske.GradientBoostingClassifier, default_gradient_boosting,
                  space_gradient_boosting_classifier, 'classification', rule_params=rule_gbm, use_predict_proba=True,
                  level=2),
    ModelSolution('STK-LOGIT', 'Stacking Logistic Regression', linear.LogisticRegression,
                  default_logistic_regression, space_logistic_regression, 'classification', use_predict_proba=True,
                  level=2, rule_params=rule_logistic),

    # ensemble regressors
    ModelSolution('STK-LGBM-R', 'Stacking LightGBM', ModelLightGBM, default_lightgbm_regressor,
                  space_lightgbm_regressor, 'regression', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_lightgbm, selectable=import_lgbm),
    ModelSolution('STK-XGB-R', 'Stacking XgBoost', ModelXgBoost, default_xgboost_regressor,
                  space_xgboost_regressor, 'regression', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_xgboost, selectable=import_xgb),
    ModelSolution('STK-NN-R', 'Stacking Neural Networks', ModelNN, default_keras,
                  space_keras, 'classification', is_wrapper=True, use_early_stopping=True, level=2,
                  rule_params=rule_nn, selectable=import_keras, limit_size=100),
    ModelSolution('STK-XTRA-R', 'Stacking Extra Trees', ske.ExtraTreesRegressor, default_extra_trees,
                  space_extra_trees_regressor, 'regression', level=2),
    ModelSolution('STK-RF-R', 'Stacking Random Forest', ske.RandomForestRegressor, default_random_forest,
                  space_random_forest_regressor, 'regression', level=2),
    ModelSolution('STK-GBM-R', 'Stacking Gradient Boosting', ske.GradientBoostingRegressor,
                  default_gradient_boosting, space_gradient_boosting_regressor, 'regression',
                  rule_params=rule_gbm, level=2),
    ModelSolution('STK-LR', 'Stacking Linear Regression', linear.LinearRegression,
                  default_linear_regression, space_linear_regression, 'regression', level=2),
]

# mapping table
model_solutions_map = {s.ref: s for s in model_solutions}
