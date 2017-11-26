from abc import ABCMeta, abstractmethod
import pickle
import logging
import numpy as np
import pandas as pd
from .spaces.model import *
from .config import METRIC_NULL
from .dataset import get_dataset_folder

log = logging.getLogger(__name__)


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

try:
    from catboost import Pool, CatBoostClassifier, CatBoostRegressor

    import_catboost = True
except:
    import_catboost = False
    log.info('could not import Catboost. This model will not be used')

try:
    from .utils.keras_wrapper import keras_create_model, keras_compile_model, import_keras, to_categorical
except:
    log.info('could not import keras. Neural networks will not be used')

MAX_ROUNDS = 5000
PATIENCE = 50


def get_importance(dataset_id, round_id):
    """
    features importance of the model

    :param dataset_id: id of the dataset
    :param round_id: id of the round
    :return: feature importance as a dataframe
    """
    try:
        return pickle.load(open(get_dataset_folder(dataset_id) + '/features/%s.pkl' % round_id, 'rb'))
    except:
        return None


def get_pred_eval_test(dataset_id, round_id):
    """
    prediction on eval set & test & submit set

    :param dataset_id: id of the dataset
    :param round_id: id of the round
    :return: list of predictions for eval set, test and submit set
    """
    return pickle.load(open(get_dataset_folder(dataset_id) + '/predict/%s.pkl' % round_id, 'rb'))


class Model(object):
    __metaclass__ = ABCMeta

    # abstract class for model hyper optimization

    @abstractmethod
    def __init__(self, dataset, context, params):
        self.dataset = dataset
        self.context = context
        self.params = params
        self.feature_names = context.feature_names


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
        return self.model.predict(X)

    @abstractmethod
    def predict_proba(self, X):
        # predict with the previously trained model
        return self.model.predict_proba(X)


def binary_proba(y):
    # convert a binary proba of 1 dimension (on true) to 2 dimensions (false, true)
    return np.stack([1 - y, y], axis=1)



class ModelLightGBM(Model):
    # class for model LightGBM

    def __init__(self, dataset, context, params):
        super().__init__(dataset, context, params)

    def fit(self, X_train, y_train):
        # train with num_rounds
        lgb_train = lgb.Dataset(X_train, y_train)
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=self.num_rounds)
        self.feature_importances_ = self.model.feature_importance()

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # specific early stopping for Light GBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.model = lgb.train(self.params,
                               lgb_train,
                               num_boost_round=MAX_ROUNDS,
                               valid_sets=lgb_eval,
                               early_stopping_rounds=PATIENCE, verbose_eval=False)
        # check early stopping
        if self.model.best_iteration == 0:
            self.num_rounds = MAX_ROUNDS
        else:
            self.num_rounds = self.model.best_iteration
            log.info('best iteration at %d' % self.model.best_iteration)

    def predict_proba(self, X):
        # prediction with specific case of binary
        if self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(X))
        else:
            return self.model.predict(X)


class ModelXgBoost(Model):
    # class for model XGBOOST

    def __init__(self, dataset, context, params):
        super().__init__(dataset, context, params)

    def fit(self, X_train, y_train):
        # train with num_rounds
        xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        self.model = xgb.train(self.params,
                               xgb_train,
                               num_boost_round=self.num_rounds)

        self.dict_importance_ = self.model.get_score()

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # specific early stopping for XxBoost
        xgb_train = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        xgb_eval = xgb.DMatrix(X_eval, label=y_eval, feature_names=self.feature_names)
        self.model = xgb.train(self.params,
                               xgb_train,
                               MAX_ROUNDS,
                               evals=[(xgb_train, 'train'), (xgb_eval, 'eval')],
                               early_stopping_rounds=PATIENCE, verbose_eval=False)

        if self.model.best_iteration > 0:
            self.num_rounds = self.model.best_iteration
        else:
            self.num_rounds = PATIENCE

    def predict(self, X):
        xgb_X = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(xgb_X)

    def predict_proba(self, X):
        # prediction with specific case of binary
        xgb_X = xgb.DMatrix(X, feature_names=self.feature_names)
        if self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(xgb_X))
        else:
            return self.model.predict(xgb_X)


class ModelCatboost(Model):
    # class for model Catboost

    def __init__(self, dataset, context, params):
        super().__init__(dataset, context, params)
        self.early_stopping = True
        self.feature_importance = []
        self.set_model()

    def set_model(self):
        # set loss function depending of binary / multi class problem
        if self.dataset.problem_type == 'regression':
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostClassifier(**self.params)

    def fit(self, X_train, y_train):
        # train with num_rounds
        train_pool = Pool(X_train, label=y_train.astype(float))
        self.set_model()
        self.model.fit(train_pool, use_best_model=False)
        self.feature_importances_ = self.model.get_feature_importance(train_pool)

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



class ModelNN(Model):
    # class for model Neural Networks

    def __init__(self, dataset, context, params):
        super().__init__(dataset, context, params)
        self.early_stopping = True
        self.model_created = False

        # create the modem
        self.params['input_dim'] = len(self.feature_names)
        self.model = keras_create_model(self.params, self.dataset.problem_type)

    def fit(self, X_train, y_train):
        # train with num_rounds
        keras_compile_model(self.model, self.params, self.dataset.problem_type)
        self.model.fit(X_train, self.prepare_y(y_train), epochs=self.num_rounds, batch_size=self.params['batch_size'],
                       validation_split=0.,
                       verbose=0)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # find best round
        keras_compile_model(self.model, self.params, self.dataset.problem_type)
        best_score = METRIC_NULL
        i_best_score = 0
        self.num_rounds = 0
        for i in range(MAX_ROUNDS):
            self.model.fit(X_train, self.prepare_y(y_train), epochs=1, batch_size=self.params['batch_size'],
                           validation_split=0.,
                           verbose=0)
            if self.dataset.problem_type == 'regression':
                y_pred = self.model.predict(X_eval)
            else:
                y_pred = self.model.predict_proba(X_eval)
            score = self.dataset.evaluate_metric(y_eval, y_pred)
            if score < best_score:
                self.num_rounds = i
                best_score = score
                i_best_score = i
            elif (i - i_best_score) > 50:
                log.info('early stopping at %d' % i_best_score)
                break

    def predict_proba(self, X):
        # prediction with specific case of binary and classification
        if self.dataset.y_n_classes == 2:
            return binary_proba(self.model.predict(X))
        else:
            return self.model.predict_proba(X)

    def prepare_y(self, y):
        # generate y in one hot encoding if classification
        if self.dataset.problem_type == 'classification':
            return to_categorical(y, self.dataset.y_n_classes)
        else:
            return y


class EnsemblePool(object):
    # class to manage data required for ensembling
    def __init__(self, pool_model_round_ids, pool_model_names, pool_eval_preds, pool_test_preds, pool_submit_preds):
        self.pool_model_round_ids = pool_model_round_ids
        self.pool_model_names = pool_model_names
        self.pool_eval_preds = pool_eval_preds
        self.pool_test_preds = pool_test_preds
        self.pool_submit_preds = pool_submit_preds


class ModelEnsembleSelection(Model):
    # TODO : fix bug
    # class for model with ensemble selection

    def __init__(self, dataset, context, params):
        super().__init__(dataset, context, params)
        self.selection = None

    def cv_pool(self, pool, y, y_test, cv_folds, threshold, depth):
        y_pred_eval_list, y_pred_test_list, y_pred_submit_list = [], [], []
        self.params = {**{'depth': depth}, **self.params}
        for i, (train_index, eval_index) in enumerate(cv_folds):
            log.info('fold %d' % i)
            # we will select a list of models in order to get the best score
            selection_round_ids, selection_names = [], []
            pred_select_eval, pred_select_test, pred_select_submit = [], [], []
            best_score = METRIC_NULL
            for i in range(self.params['rounds']):
                # log.info('round %d' % i)
                # find the best model to be added in the selection
                best_score_round = METRIC_NULL
                l_selection = len(selection_round_ids)
                for u, m, p_eval, p_test, p_submit in zip(pool.pool_model_round_ids, pool.pool_model_names,
                                                          pool.pool_eval_preds, pool.pool_test_preds,
                                                          pool.pool_submit_preds):
                    # prediction = weighted average of predictions
                    # try:
                    if l_selection < 1:
                        y_pred_eval = p_eval
                        y_pred_test = p_test
                        y_pred_submit = p_submit
                    else:
                        y_pred_eval = (pred_select_eval * l_selection + p_eval) / (l_selection + 1)
                        y_pred_test = (pred_select_test * l_selection + p_test) / (l_selection + 1)
                        y_pred_submit = (pred_select_submit * l_selection + p_submit) / (l_selection + 1)
                    if np.shape(y[train_index]) == np.shape(y_pred_eval[train_index]):
                        score = self.dataset.evaluate_metric(y[train_index], y_pred_eval[train_index])
                    else:
                        score = METRIC_NULL
                    # except:
                    #    score = METRIC_NULL

                    if score < best_score_round:
                        # log.info('best score round', m, score)
                        best_score_round = score
                        m_round, u_round = m, u
                        pred_round_eval, pred_round_test, pred_round_submit = y_pred_eval, y_pred_test, y_pred_submit

                # at the end of the search for the round, check if the overall score is better
                if best_score_round < best_score:
                    log.info('best score:', best_score_round)
                    best_score = best_score_round
                    pred_select_eval, pred_select_test, pred_select_submit = pred_round_eval, pred_round_test, pred_round_submit
                    selection_names += [m_round]
                    selection_round_ids += [u_round]
                else:
                    # didn't improve = early stopping
                    break

            log.info(np.shape(pred_select_eval))
            log.info(np.shape(eval_index))
            y_pred_eval_list.append(pred_select_eval[eval_index])
            y_pred_test_list.append(pred_select_test)
            y_pred_submit_list.append(pred_select_submit)

        # calculate weights for the models
        df = pd.DataFrame([(round_id, name, 1) for round_id, name in zip(selection_round_ids, selection_names)])
        df.columns = ['round_id', 'name', 'weight']
        self.selection = df.groupby(['round_id', 'name'], as_index=False).sum()

        # then create feature importance with the names of the models
        self.importance = self.selection[['name', 'weight']]
        self.importance.columns = ['feature', 'importance']

        # average pred_test
        y_pred_test = np.mean(y_pred_test_list)

        return False, y_pred_eval_list, y_pred_test, y_pred_submit_list
