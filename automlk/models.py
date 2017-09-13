from abc import ABCMeta, abstractmethod
import pickle
import uuid
import time
import sklearn.ensemble as ske
import sklearn.linear_model as linear
import sklearn.svm as svm
import sklearn.neighbors as knn
import numpy as np
import pandas as pd

from .spaces.model import *
from .spaces.hyper import get_random_params
from .dataset import METRIC_NULL, get_dataset_folder
from .utils.keras_wrapper import keras_create_model, keras_compile_model, import_keras

# TODO: add naives bayes and decision trees

try:
    import lightgbm as lgb

    import_lgbm = True
except:
    import_lgbm = False
    print('could not import LightGBM. This model will not be used')

try:
    import xgboost as xgb

    import_xgb = True
except:
    import_xgb = False
    print('could not import Xgboost. This model will not be used')

try:
    from catboost import Pool, CatBoostClassifier, CatBoostRegressor

    import_catboost = True
except:
    import_catboost = False
    print('could not import Catboost. This model will not be used')


MAX_ROUNDS = 5000
PATIENCE = 50


def get_importance(uid, uuid):
    # return stored features importance
    try:
        return pickle.load(open(get_dataset_folder(uid) + '/features/%s.pkl' % uuid, 'rb'))
    except:
        return None


def get_pred_eval_test(uid, uuid):
    # return stored prediction on eval set & test set
    return pickle.load(open(get_dataset_folder(uid) + '/predict/%s.pkl' % uuid, 'rb'))


class HyperModel(object):
    __metaclass__ = ABCMeta

    # abstract class for model hyper optimization

    @abstractmethod
    def __init__(self, dataset, context):
        self.dataset = dataset
        self.context = context
        self.feature_names = context.feature_names
        self.params = {}
        self.uuid = str(uuid.uuid4())
        self.model = None
        self.num_rounds = 0
        self.model_loaded = True
        self.y_pred_eval = None
        self.model_level = 0
        self.early_stopping = False
        self.importance = None

    @abstractmethod
    def set_default_params(self):
        # generate default parameters of the model
        pass

    @abstractmethod
    def set_random_params(self):
        # generate random parameters of the model on a specific subspace
        pass

    @abstractmethod
    def set_model(self):
        # pre-generates the model
        pass

    @abstractmethod
    def cv(self, X, y, X_test, y_test, cv_folds, threshold):
        # performs a cross validation on cv_folds, and predict also on X_test
        y_pred_eval, y_pred_test = [], []
        for i, (train_index, eval_index) in enumerate(cv_folds):
            if i == 0 and self.early_stopping:
                print('early stopping round')
                # with early stopping, we perform an initial round to get number of rounds
                self.fit_early_stopping(X[train_index], y[train_index], X[eval_index], y[eval_index])

                if threshold != 0:
                    # test outlier (i.e. exceeds threshold)
                    y_pred = self.predict(X[eval_index])
                    score = self.dataset.evaluate_metric(y[eval_index], y_pred)
                    print('early stopping score: %.5f' % score)
                    if score > threshold:
                        print('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
                        time.sleep(10)
                        return True, y_pred_eval, y_pred_test

            # then train on train set and predict on eval set
            self.fit(X[train_index], y[train_index])
            y_pred = self.predict(X[eval_index])

            if threshold != 0:
                # test outlier:
                score = self.dataset.evaluate_metric(y[eval_index], y_pred)
                print('fold %d score: %.5f' % (i, score))
                if score > threshold:
                    print('%dth round found outlier: %.5f with threshold %.5f' % (i, score, threshold))
                    time.sleep(10)
                    return True, y_pred_eval, y_pred_test

            y_pred_eval.append(y_pred)

            # we also predict on test set (to be averaged later)
            y_pred_test.append(self.predict(X_test))

        return False, y_pred_eval, y_pred_test

    @abstractmethod
    def fit(self, X_train, y_train):
        # fits the model on X_train and y_train
        self.model.fit(X_train, y_train)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # fits the model on X_train and y_train, with test on eval in order to determine self.num_rounds
        pass

    @abstractmethod
    def predict(self, X):
        # predict with the previously trained model
        if self.dataset.problem_type == 'regression':
            return self.model.predict(X)
        else:
            # for classification, prediction is the proba
            return self.model.predict_proba(X)

    @abstractmethod
    def save_model(self):
        # saves parameters of the model
        pickle.dump([self.model], open(get_dataset_folder(self.dataset.uid) + '/models/%s.pkl' % self.uuid, 'wb'))

    @abstractmethod
    def save_importance(self):
        # saves feature importance (as a dataframe)
        if hasattr(self.model, 'feature_importances_'):
            self.importance = pd.DataFrame(self.feature_names)
            self.importance['importance'] = self.model.feature_importances_
            self.importance.columns = ['feature', 'importance']
            pickle.dump(self.importance, open(self.feature_filename(), 'wb'))

    @abstractmethod
    def feature_filename(self):
        # filename for feature importance
        return get_dataset_folder(self.dataset.uid) + '/features/%s.pkl' % self.uuid

    @abstractmethod
    def save_predict(self, y_pred_eval, y_pred_test):
        # save predictions (eval and test set)
        pickle.dump([y_pred_eval, y_pred_test],
                    open(get_dataset_folder(self.dataset.uid) + '/predict/%s.pkl' % self.uuid, 'wb'))


def binary_proba(y):
    # convert a binary proba of 1 dimension (on true) to 2 dimensions (false, true)
    return np.stack([1 - y, y], axis=1)


class HyperModelLogisticRegression(HyperModel):
    # class for model Logistic regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Logistic Regression'

    def set_default_params(self):
        self.params = default_logistic_regression

    def set_random_params(self):
        self.params = get_random_params(space_logistic_regression)
        if self.params['dual']:
            self.params['penalty'] = 'l2'

    def set_model(self):
        self.model = linear.LogisticRegression(**self.params)


class HyperModelLinearRegressor(HyperModel):
    # class for model Linear regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Linear Regression'

    def set_default_params(self):
        self.params = default_linear_regression

    def set_random_params(self):
        self.params = get_random_params(space_linear_regression)

    def set_model(self):
        self.model = linear.LinearRegression(**self.params)


class HyperModelLassoRegressor(HyperModel):
    # class for model Lasso Linear regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Lasso Regression'

    def set_default_params(self):
        self.params = default_lasso_regression

    def set_random_params(self):
        self.params = get_random_params(space_lasso_regression)

    def set_model(self):
        self.model = linear.Lasso(**self.params)


class HyperModelRidgeRegressor(HyperModel):
    # class for model Ridge Linear regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Ridge Regression'

    def set_default_params(self):
        self.params = default_ridge_regression

    def set_random_params(self):
        self.params = get_random_params(space_ridge_regression)

    def set_model(self):
        self.model = linear.Ridge(**self.params)


class HyperModelHuberRegressor(HyperModel):
    # class for model Huber Linear regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Huber Regression'

    def set_default_params(self):
        self.params = default_huber_regression

    def set_random_params(self):
        self.params = get_random_params(space_huber_regression)

    def set_model(self):
        self.model = linear.HuberRegressor(**self.params)


class HyperModelLinearSVC(HyperModel):
    # class for model SVM classification

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'SVM Linear'

    def set_default_params(self):
        self.params = default_linear_svc

    def set_random_params(self):
        self.params = get_random_params(space_linear_svc)

    def set_model(self):
        self.model = svm.LinearSVC(**self.params)


class HyperModelLinearSVR(HyperModel):
    # class for model SVM regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'SVM Linear'

    def set_default_params(self):
        self.params = default_linear_svc

    def set_random_params(self):
        self.params = get_random_params(space_linear_svr)

    def set_model(self):
        self.model = svm.LinearSVR(**self.params)


class HyperModelSVM(HyperModel):
    # class for model SVM kernel

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'SVM kernel'

    def set_default_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = default_svr
        else:
            self.params = default_svc

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_svr)
        else:
            self.params = get_random_params(space_svc)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = svm.SVR(**self.params)
        else:
            self.model = svm.SVC(**self.params)


class HyperModelKnn(HyperModel):
    # class for model KNN

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'KNN'

    def set_default_params(self):
        self.params = default_knn

    def set_random_params(self):
        self.params = get_random_params(space_knn)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = knn.KNeighborsRegressor(**self.params)
        else:
            self.model = knn.KNeighborsClassifier(**self.params)


class HyperModelAdaBoost(HyperModel):
    # class for model AdaBoost

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'AdaBoost'

    def set_default_params(self):
        self.params = default_adaboost

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_adaboost_regressor)
        else:
            self.params = get_random_params(space_adaboost_classifier)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = ske.AdaBoostRegressor(**self.params)
        else:
            self.model = ske.AdaBoostClassifier(**self.params)


class HyperModelGradientBoosting(HyperModel):
    # class for model AdaBoost

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Gradient Boosting'

    def set_default_params(self):
        self.params = default_gradient_boosting

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_gradient_boosting_regressor)
        else:
            self.params = get_random_params(space_gradient_boosting_classifier)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = ske.GradientBoostingRegressor(**self.params)
        else:
            self.model = ske.GradientBoostingClassifier(**self.params)


class HyperModelExtraTrees(HyperModel):
    # class for model Extra trees

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Extra Trees'

    def set_default_params(self):
        self.params = default_extra_trees

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_extra_trees_regressor)
        else:
            self.params = get_random_params(space_extra_trees_classifier)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = ske.ExtraTreesRegressor(**self.params)
        else:
            self.model = ske.ExtraTreesClassifier(**self.params)


class HyperModelRandomForest(HyperModel):
    # class for model Random Forest

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Random Forest'

    def set_default_params(self):
        self.params = default_random_forest

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_random_forest_regressor)
        else:
            self.params = get_random_params(space_random_forest_classifier)

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = ske.RandomForestRegressor(**self.params)
        else:
            self.model = ske.RandomForestClassifier(**self.params)


class HyperModelLightGBM(HyperModel):
    # class for model LightGBM

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Light GBM'
        self.model_loaded = import_lgbm
        self.early_stopping = True

    def set_default_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = default_lightgbm_regressor
        elif self.dataset.y_n_classes == 2:
            self.params = default_lightgbm_binary
        else:
            self.params = default_lightgbm_classifier
            self.params['num_class'] = self.dataset.y_n_classes

    def set_random_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_lightgbm_regressor)
        elif self.dataset.y_n_classes == 2:
            self.params = get_random_params(space_lightgbm_binary)
        else:
            self.params = get_random_params(space_lightgbm_classifier)
            self.params['num_class'] = self.dataset.y_n_classes
        self.valid_params()

    def valid_params(self):
        # updates params according to Light GBM rules
        if self.params['bagging_freq'] == 0:
            self.params.pop('bagging_freq', None)
        if self.params['boosting'] == 'goss':
            self.params.pop('bagging_freq', None)
            self.params.pop('bagging_fraction', None)

    def fit(self, X_train, y_train):
        # train with num_rounds
        lgb_train = lgb.Dataset(X_train, y_train)
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=self.num_rounds)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # specific early stopping for Light GBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=MAX_ROUNDS,
                               valid_sets=lgb_eval,
                               early_stopping_rounds=PATIENCE)
        # check early stopping
        if self.model.best_iteration == 0:
            self.num_rounds = MAX_ROUNDS
        else:
            self.num_rounds = self.model.best_iteration
            print('best iteration at', self.model.best_iteration)

    def predict(self, X):
        # prediction with specific case of binary
        if self.dataset.problem_type == 'regression':
            return self.model.predict(X)
        elif self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(X))
        else:
            return self.model.predict(X)

    def save_importance(self):
        # saves feature importance (as a dataframe)
        self.importance = pd.DataFrame(self.feature_names)
        self.importance['importance'] = self.model.feature_importance()
        self.importance.columns = ['feature', 'importance']
        pickle.dump(self.importance, open(self.feature_filename(), 'wb'))


class HyperModelXgBoost(HyperModel):
    # class for model XGBOOST

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Xgboost'
        self.model_loaded = import_xgb
        self.early_stopping = True

    def set_default_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = default_xgboost_regressor
        elif self.dataset.y_n_classes == 2:
            self.params = default_xgboost_binary
        else:
            self.params = default_xgboost_classifier
            self.params['num_class'] = self.dataset.y_n_classes

    def set_random_params(self):
        # generate params
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_xgboost_regressor)
        elif self.dataset.y_n_classes == 2:
            self.params = get_random_params(space_xgboost_binary)
        else:
            self.params = get_random_params(space_xgboost_classifier)
            self.params['num_class'] = self.dataset.y_n_classes

    def fit(self, X_train, y_train):
        # train with num_rounds
        xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        self.model = xgb.train(self.params,
                               xgb_train,
                               num_boost_round=self.num_rounds)


    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # specific early stopping for XxBoost
        xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        xgb_eval = xgb.DMatrix(X_eval, label=y_eval, feature_names=self.feature_names)
        self.model = xgb.train(self.params,
                               xgb_train,
                               MAX_ROUNDS,
                               evals=[(xgb_train, 'train'), (xgb_eval, 'eval')],
                               early_stopping_rounds=PATIENCE)

        if self.model.best_iteration > 0:
            self.num_rounds = self.model.best_iteration
        else:
            self.num_rounds = PATIENCE


    def predict(self, X):
        # prediction with specific case of binary
        xgb_X = xgb.DMatrix(X, feature_names=self.feature_names)
        if self.dataset.problem_type == 'regression':
            return self.model.predict(xgb_X)
        elif self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(xgb_X))
        else:
            return self.model.predict(xgb_X)

    def save_importance(self):
        # saves feature importance (as a dataframe)
        imp = self.model.get_score()
        importance = pd.DataFrame([{'feature': key, 'importance': imp[key]} for key in imp.keys()])
        pickle.dump(importance, open(self.feature_filename(), 'wb'))


class HyperModelCatboost(HyperModel):
    # class for model Catboost

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Catboost'
        self.model_loaded = import_catboost
        self.early_stopping = True
        self.feature_importance = []

    def set_default_params(self):
        if self.dataset.problem_type == 'regression':
            self.params = default_catboost_regressor
        else:
            self.params = default_catboost_classifier
            self.__set_loss()

    def set_random_params(self):
        # generate params
        if self.dataset.problem_type == 'regression':
            self.params = get_random_params(space_catboost_regressor)
        else:
            self.params = get_random_params(space_catboost_classifier)
            self.__set_loss()

    def __set_loss(self):
        # set loss function depending of binary / multi class problem
        if self.dataset.y_n_classes == 2:
            self.params['loss_function'] = 'Logloss'
        else:
            self.params['loss_function'] = 'MultiClass'

    def set_model(self):
        if self.dataset.problem_type == 'regression':
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostClassifier(**self.params)

    def fit(self, X_train, y_train):
        # train with num_rounds
        train_pool = Pool(X_train, label=y_train.astype(float))
        self.set_model()
        self.model.fit(train_pool, use_best_model=False)
        self.feature_importance = self.model.get_feature_importance(train_pool)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # specific early stopping for Catboost
        train_pool = Pool(X_train, label=y_train.astype(float))
        eval_pool = Pool(X_eval, label=y_eval.astype(float))
        # set specific parameters for early stopping (overfitting detector with iter)
        self.params['iterations'] = MAX_ROUNDS
        self.params['od_type'] = 'iter'
        self.params['od_wait'] = PATIENCE

        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        self.num_rounds = self.model.tree_count_

        self.params['iterations'] = self.num_rounds
        self.params.pop('od_type')
        self.params.pop('od_wait')

    def predict(self, X):
        # prediction with specific case of binary
        if self.dataset.problem_type == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)

    def save_importance(self):
        # saves feature importance (as a dataframe)
        self.importance = pd.DataFrame(self.feature_names)
        self.importance['importance'] = self.feature_importance
        self.importance.columns = ['feature', 'importance']
        pickle.dump(self.importance, open(self.feature_filename(), 'wb'))


class HyperModelNN(HyperModel):
    # class for model Neural Networks

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Neural Networks'
        self.model_loaded = import_keras
        self.early_stopping = True
        self.model_created = False

    def set_default_params(self):
        self.params = default_keras

    def set_random_params(self):
        self.params = get_random_params(space_keras)

    def set_model(self):
        # create the modem
        self.params['input_dim'] = len(self.feature_names)
        if self.dataset.problem_type == 'regression':
            self.params['output_dim'] = 1
        else:
            self.params['output_dim'] = self.dataset.y_n_classes
        self.model = keras_create_model(self.params, self.dataset.problem_type)

    def fit(self, X_train, y_train):
        # train with num_rounds
        keras_compile_model(self.model, self.params, self.dataset.problem_type)
        self.model.fit(X_train, y_train, epochs=self.num_rounds, batch_size=self.params['batch_size'],
                       validation_split=0.,
                       verbose=0)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # find best round
        best_score = METRIC_NULL
        i_best_score = 0
        for i in range(MAX_ROUNDS):
            self.model.fit(X_train, y_train, epochs=1, batch_size=self.params['batch_size'],
                           validation_split=0.,
                           verbose=0)

            y_pred = self.model.predict(X_eval)
            score = self.dataset.evaluate_metric(y_eval, y_pred)
            if score < best_score:
                self.num_rounds = i
                best_score = score
                i_best_score = i
            elif (i - i_best_score) > 50:
                print('early stopping at', i_best_score)
                break

    def predict(self, X):
        # prediction with specific case of binary and classification
        if self.dataset.problem_type == 'regression':
            return self.model.predict(X)
        elif self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(X))
        else:
            return self.model.predict_proba(X)

    def save_model(self):
        # TODO: implement save weights
        pass
        # pickle.dump([self.model], open(dataset_root_folder(self.dataset.uid) + '/models/model_%s.pkl' % self.uuid, 'wb'))


class EnsemblePool(object):
    # class to manage data required for ensembling
    def __init__(self, pool_model_uuids, pool_model_names, pool_eval_preds, pool_test_preds):
        self.pool_model_uuids = pool_model_uuids
        self.pool_model_names = pool_model_names
        self.pool_eval_preds = pool_eval_preds
        self.pool_test_preds = pool_test_preds


class HyperModelEnsembleSelection(HyperModel):
    # class for model with ensemble selection

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Ensemble Selection'
        self.model_level = 1
        self.params = {}
        self.selection = None

    def set_default_params(self):
        self.params = default_ensemble_selection

    def set_random_params(self):
        self.params = get_random_params(space_ensemble_selection)

    def cv_pool(self, pool, y, y_test, cv_folds, threshold, depth):
        y_pred_eval_list = []
        y_pred_test_list = []
        self.params = {**{'depth': depth}, **self.params}
        for i, (train_index, eval_index) in enumerate(cv_folds):
            print('fold %d' % i)
            # we will select a list of models in order to get the best score
            selection_uuids, selection_names, pred_select_eval, pred_select_test = [], [], [], []
            best_score = METRIC_NULL
            for i in range(self.params['rounds']):
                # print('round %d' % i)
                # find the best model to be added in the selection
                best_score_round = METRIC_NULL
                l_selection = len(selection_uuids)
                for u, m, p_eval, p_test in zip(pool.pool_model_uuids, pool.pool_model_names, pool.pool_eval_preds,
                                                pool.pool_test_preds):
                    # prediction = weighted average of predictions
                    try:
                        if l_selection < 1:
                            y_pred_eval = p_eval
                            y_pred_test = p_test
                        else:
                            y_pred_eval = (pred_select_eval * l_selection + p_eval) / (l_selection + 1)
                            y_pred_test = (pred_select_test * l_selection + p_test) / (l_selection + 1)
                        if np.shape(y[train_index]) == np.shape(y_pred_eval[train_index]):
                            score = self.dataset.evaluate_metric(y[train_index], y_pred_eval[train_index])
                        else:
                            score = METRIC_NULL
                    except:
                        score = METRIC_NULL
                    if score < best_score_round:
                        # print('best score round', m, score)
                        best_score_round = score
                        m_round, u_round = m, u
                        pred_round_eval, pred_round_test = y_pred_eval, y_pred_test
                # at the end of the search for the round, check if the overall score is better
                if best_score_round < best_score:
                    # print('best score:', best_score_round)
                    best_score = best_score_round
                    pred_select_eval, pred_select_test = pred_round_eval, pred_round_test
                    selection_names += [m_round]
                    selection_uuids += [u_round]
                else:
                    # didn't improve = early stopping
                    break

            print(np.shape(pred_select_eval))
            print(np.shape(eval_index))
            y_pred_eval_list.append(pred_select_eval[eval_index])
            y_pred_test_list.append(pred_select_test)

        # calculate weights for the models
        df = pd.DataFrame([(uuid, name, 1) for uuid, name in zip(selection_uuids, selection_names)])
        df.columns = ['uuid', 'name', 'weight']
        self.selection = df.groupby(['uuid', 'name'], as_index=False).sum()

        # then create feature importance with the names of the models
        self.importance = self.selection[['name', 'weight']]
        self.importance.columns = ['feature', 'importance']

        return True, y_pred_eval_list, y_pred_test_list

    def save_importance(self):
        # saves feature importance (as a dataframe)
        pickle.dump(self.importance, open(self.feature_filename(), 'wb'))


class HyperModelStacking(HyperModel):
    # class for model with ensemble stacking

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_level = 1

    def set_default_params(self):
        self.model.set_default_params()

    def set_random_params(self):
        self.model.set_random_params()
        self.params = self.model.params

    def set_model(self):
        self.model.set_model()
        self.params = self.model.params

    @abstractmethod
    def cv_pool(self, pool, y, y_test, cv_folds, threshold, depth):
        # TODO: select a subset of the pool (best models)
        y_pred_eval = []
        y_pred_test = []
        self.params = {**{'depth': depth}, **self.params}
        # set feature names
        if self.dataset.problem_type == 'regression':
            self.feature_names = [name+'_'+uuid[:8] for uuid, name in zip(pool.pool_model_uuids, pool.pool_model_names)]
        else:
            self.feature_names = []
            for uuid, name in zip(pool.pool_model_uuids, pool.pool_model_names):
                for k in range(self.dataset.y_n_classes):
                    self.feature_names.append(name+'_'+str(k)+'_'+uuid[:8])
        self.model.feature_names = self.feature_names

        for i, (train_index, eval_index) in enumerate(cv_folds):
            print('fold %d' % i)
            # create X by stacking predictions
            for j, (u, m, p_eval, p_test) in enumerate(
                    zip(pool.pool_model_uuids, pool.pool_model_names, pool.pool_eval_preds,
                        pool.pool_test_preds)):
                # check if array has 2 dimensions
                shape = len(np.shape(p_eval))
                if shape == 1:
                    p_eval = np.reshape(p_eval, (len(p_eval), 1))
                    p_test = np.reshape(p_test, (len(p_test), 1))
                if j == 0:
                    X_train = p_eval[train_index]
                    X_eval = p_eval[eval_index]
                    X_test = p_test
                else:
                    # stack vertically the predictions
                    X_train = np.concatenate((X_train, p_eval[train_index]), axis=1)
                    X_eval = np.concatenate((X_eval, p_eval[eval_index]), axis=1)
                    X_test = np.concatenate((X_test, p_test), axis=1)

            if i == 0 and self.model.early_stopping:
                # with early stopping, we perform an initial round to get number of rounds
                print('fit early stopping')
                self.model.fit_early_stopping(X_train, y[train_index], X_eval, y[eval_index])
                y_pred = self.model.predict(X_eval)
                score = self.dataset.evaluate_metric(y[eval_index], y_pred)
                print('early stopping score: %.5f' % score)
                if score > threshold:
                    print('early stopping found outlier: %.5f with threshold %.5f' % (score, threshold))
                    time.sleep(10)
                    return True, y_pred_eval, y_pred_test

            # train on X_train
            self.model.fit(X_train, y[train_index])
            y_pred = self.model.predict(X_eval)
            y_pred_eval.append(y_pred)
            y_pred_test.append(self.model.predict(X_test))
            score = self.dataset.evaluate_metric(y[eval_index], y_pred)
            if score > threshold:
                print('found outlier: %.5f with threshold %.5f' % (score, threshold))
                time.sleep(10)
                return True, y_pred_eval, y_pred_test

        return False, y_pred_eval, y_pred_test

    # TODO: implement predict
    # TODO: implement load/save

    def save_importance(self):
        self.model.save_importance()
        if isinstance(self.model.importance, pd.DataFrame):
            self.importance = self.model.importance
            pickle.dump(self.importance, open(self.feature_filename(), 'wb'))


class HyperModelStackingExtraTrees(HyperModelStacking):
    # class for stacking with model Extra trees

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking Extra Trees'
        self.model = HyperModelExtraTrees(dataset, context)


class HyperModelStackingRandomForest(HyperModelStacking):
    # class for stacking with model Extra trees

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking Random Forest'
        self.model = HyperModelRandomForest(dataset, context)


class HyperModelStackingGradientBoosting(HyperModelStacking):
    # class for stacking with model Extra trees

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking Gradient Boosting'
        self.model = HyperModelGradientBoosting(dataset, context)


class HyperModelStackingLinear(HyperModelStacking):
    # class for stacking with model Extra trees

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking Linear'
        self.model = HyperModelLinearRegressor(dataset, context)

class HyperModelStackingLogistic(HyperModelStacking):
    # class for stacking with model Logistic regression

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking Logistic'
        self.model = HyperModelLogisticRegression(dataset, context)

class HyperModelStackingXgBoost(HyperModelStacking):
    # class for stacking with model XgBoost

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking XgBoost'
        self.model = HyperModelXgBoost(dataset, context)

class HyperModelStackingLightGBM(HyperModelStacking):
    # class for stacking with model LightGBM

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking LightGBM'
        self.model = HyperModelLightGBM(dataset, context)

class HyperModelStackingNN(HyperModelStacking):
    # class for stacking with model NN

    def __init__(self, dataset, context):
        super().__init__(dataset, context)
        self.model_name = 'Stacking NN'
        self.model = HyperModelNN(dataset, context)
