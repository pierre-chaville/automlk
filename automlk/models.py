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


class Model(object):
    __metaclass__ = ABCMeta

    # abstract class for model hyper optimization

    @abstractmethod
    def __init__(self, **params):
        self.params = params
        self.problem_type = params['problem_type']
        self.y_n_classes = params['y_n_classes']
        self.model_params = {key: params[key] for key in params.keys() if key not in ['problem_type', 'y_n_classes']}

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


class ModelCatboost(Model):
    # class for model Catboost

    def __init__(self, **params):
        super().__init__(**params)
        self.early_stopping = True
        self.feature_importance = []
        self.set_model()

    def set_model(self):
        # set loss function depending of binary / multi class problem
        if self.problem_type == 'regression':
            self.model = CatBoostRegressor(**self.model_params)
        else:
            self.model = CatBoostClassifier(**self.model_params)

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

    def __init__(self, **params):
        super().__init__(**params)
        self.early_stopping = True
        self.model_created = False

    def fit(self, X_train, y_train):
        # train with num_rounds
        if not self.model_created:
            self.__create_model(len(X_train.columns))
        keras_compile_model(self.model, self.params, self.problem_type)
        self.model.fit(X_train.as_matrix(), self.prepare_y(y_train), epochs=self.num_rounds, batch_size=self.params['batch_size'],
                       validation_split=0.,
                       verbose=0)

    def fit_early_stopping(self, X_train, y_train, X_eval, y_eval):
        # find best round
        if not self.model_created:
            self.__create_model(len(X_train.columns))
        keras_compile_model(self.model, self.params, self.problem_type)
        best_score = METRIC_NULL
        i_best_score = 0
        self.num_rounds = 0
        for i in range(MAX_ROUNDS):
            self.model.fit(X_train.as_matrix(), self.prepare_y(y_train), epochs=1, batch_size=self.params['batch_size'],
                           validation_split=0.,
                           verbose=0)
            if self.problem_type == 'regression':
                y_pred = self.model.predict(X_eval.as_matrix())
            else:
                y_pred = self.model.predict_proba(X_eval.as_matrix())
            score = self.dataset.evaluate_metric(y_eval, y_pred)
            if score < best_score:
                self.num_rounds = i
                best_score = score
                i_best_score = i
            elif (i - i_best_score) > 50:
                log.info('early stopping at %d' % i_best_score)
                break

    def __create_model(self, dim):
        # create the modem
        self.params['input_dim'] = dim
        self.model = keras_create_model(self.params, self.problem_type)

    def predict_proba(self, X):
        # prediction with specific case of binary and classification
        if self.y_n_classes == 2:
            return binary_proba(self.model.predict(X.as_matrix()))
        else:
            return self.model.predict_proba(X.as_matrix())

    def prepare_y(self, y):
        # generate y in one hot encoding if classification
        if self.problem_type == 'classification':
            return to_categorical(y, self.y_n_classes)
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
