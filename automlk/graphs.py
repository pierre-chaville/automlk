import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn.apionly as sns
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from .config import METRIC_NULL
from .context import get_dataset_folder, get_config


TRANSPARENT = False


def graph_histogram(dataset_id, col, is_categorical, values, part='train'):
    """
    generate the histogram of column col of the dataset
    :param dataset_id: dataset id
    :param col: column name
    :param is_categorical: is the column categorical
    :param values: values of the column
    :param part: set (train, test)
    :return: None
    """
    if is_categorical:
        df = pd.DataFrame(values)
        df.columns = ['y']
        df.fillna('', inplace=True)
        encoder = LabelEncoder()
        df['y'] = encoder.fit_transform(df['y'])
        values = df['y'].values
        # TODO: x axis ticks with labels

    plt.figure(figsize=(6, 6))
    __set_graph_style()
    plt.hist(values, label='actuals', bins=100)
    plt.title('histogram of %s (%s set)' % (col, part))
    plt.xlabel('values')
    plt.ylabel('frequencies')
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/_hist_%s_%s.png' % (part, col), transparent=TRANSPARENT)


def graph_correl_features(dataset, df):
    """
    generates the graph of correlated features (heatmap matrix)
    :param dataset: dataset object
    :param df: data (as a dataframe)
    :return: None
    """
    # convert categorical to numerical
    for col in dataset.cat_cols:
        df[col].fillna('', inplace=True).map(str)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # create correlation matrix with pandas
    corr = df.corr()

    # display heatmap
    __set_graph_style()
    if dataset.n_cols > 50:
        plt.figure(figsize=(14, 14))
    elif dataset.n_cols > 20:
        plt.figure(figsize=(10, 10))
    elif dataset.n_cols > 10:
        plt.figure(figsize=(8, 8))
    else:
        plt.figure(figsize=(6, 6))
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)

    plt.title('correlation map of the features')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(get_dataset_folder(dataset.dataset_id) + '/graphs/_correl.png', transparent=TRANSPARENT)


def __get_best_scores(scores):
    # returns the list of best scores over time
    # we will generate a list of the best values values over time
    best_scores = []
    best = METRIC_NULL
    for x in scores:
        if x < best:
            best = x
            best_scores.append(x)
        else:
            best_scores.append(best)
    return np.abs(best_scores)


def graph_history_search(dataset, df_search, best_models, level):
    """
    creates a graph of the best scores along searches
    :param dataset: dataset object
    :param df_search: dataframe of the search history
    :param best_models: selection within df_search with best models
    :param level: model level (0: standard, 1: ensembles)
    :return: None
    """
    #
    if len(df_search) < 1:
        return
    scores = df_search[df_search.level == level].sort_index().cv_max.values

    if dataset.best_is_min:
        # positive scores (e.g loss or error: min is best)
        y_lim1 = best_models.cv_max.abs().min()
        y_lim2 = best_models.cv_max.abs().quantile(0.5)
        y_lim1 -= (y_lim2 - y_lim1) * 0.02
    else:
        # negative scores (e.g. auc: max is best)
        y_lim1 = best_models.cv_max.abs().quantile(0.5)
        y_lim2 = best_models.cv_max.abs().max()
        y_lim2 += (y_lim2 - y_lim1) * 0.02

    best_scores = __get_best_scores(scores)
    plt.figure(figsize=(6, 6))
    __set_graph_style()

    plt.plot(list(range(len(best_scores))), best_scores)
    plt.title('best score over time (level=%d)' % level)
    plt.xlabel('total searches')
    plt.ylabel('score')
    plt.ylim(y_lim1, y_lim2)
    plt.savefig(get_dataset_folder(dataset.dataset_id) + '/graphs/_history_%d.png' % level, transparent=TRANSPARENT)

    # we will generate a list of the max values values per model

    plt.figure(figsize=(6, 6))
    for model_name in best_models.model_name.unique()[:5][::-1]:
        #scores = np.sort(np.abs(df_search[df_search.model_name == model_name].cv_max.values))[::-1]
        best_scores = __get_best_scores(df_search[df_search.model_name == model_name].cv_max.values)
        plt.plot(list(range(len(best_scores))), best_scores, label=model_name)

    plt.title('best score for 5 best models (level=%d)' % level)
    plt.xlabel('searches')
    plt.ylabel('score')
    plt.ylim(y_lim1, y_lim2)
    plt.legend(loc=1)
    plt.savefig(get_dataset_folder(dataset.dataset_id) + '/graphs/_models_%d.png' % level, transparent=TRANSPARENT)


def graph_predict(dataset, round_id, y, y_pred, part='eval'):
    """
    generate a graph prediction versus actuals (regression) or a confusion matrix (classification)
    :param dataset: dataset object
    :param round_id: id of the round
    :param y: actual values
    :param y_pred: predicted values
    :param part: part of the dataset
    :return: None
    """
    if dataset.problem_type == 'regression':
        plt.figure(figsize=(6, 6))
        __set_graph_style()

        # plot a graph prediction versus actuals
        plt.scatter(y, y_pred, alpha=0.2, s=2)
        mn = min(min(y), min(y_pred))
        mx = max(max(y), max(y_pred))
        plt.plot((mn, mx), (mn, mx), color='r', lw=0.7)
        plt.xlim(mn, mx)
        plt.ylim(mn, mx)
        plt.xlabel('actuals')
        plt.ylabel('predict')
        plt.title('predict versus actuals (%s set)' % part)
    else:
        plt.figure(figsize=(8, 6))
        __set_graph_style()

        # plot a confusion matrix
        normalize = False

        # convert proba to classes
        y_pred_class = np.argmax(y_pred, axis=1)

        cnf_matrix = confusion_matrix(y, y_pred_class)
        np.set_printoptions(precision=2)
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()

        tick_marks = np.arange(dataset.y_n_classes)
        plt.xticks(tick_marks, dataset.y_class_names, rotation=45)
        plt.yticks(tick_marks, dataset.y_class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('confusion matrix (%s set)' % part)

    plt.savefig(get_dataset_folder(dataset.dataset_id) + '/graphs/predict_%s_%s.png' % (part, round_id), transparent=TRANSPARENT)


def graph_pred_histogram(dataset_id, round_id, y, part='eval'):
    """
    generate the histograph of predictions
    :param dataset_id: id of the dataset
    :param round_id: id of the round (model)
    :param y: prediction values
    :param part: set (eval / train set)
    :return: None
    """
    plt.figure(figsize=(6, 6))
    __set_graph_style()
    plt.hist(y, label='prediction', bins=100)
    plt.title('histogram of predictions (%s set)' % part)
    plt.xlabel('values')
    plt.ylabel('frequencies')
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/hist_%s_%s.png' % (part, round_id), transparent=TRANSPARENT)


def __set_graph_style():
    # set the graph style according to config
    plt.style.use(get_config()['graph_style'])