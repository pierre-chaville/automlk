import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn.apionly as sns
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from .context import METRIC_NULL, get_dataset_folder


def graph_histogram(dataset_uid, col, is_categorical, values, part='train'):
    """
    generate the histogram of column col of the dataset
    :param dataset_uid: dataset id
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
    plt.style.use('dark_background')
    plt.hist(values, label='actuals', bins=100)
    plt.title('histogram of %s (%s set)' % (col, part))
    plt.xlabel('values')
    plt.ylabel('frequencies')
    plt.savefig(get_dataset_folder(dataset_uid) + '/graphs/_hist_%s_%s.png' % (part, col), transparent=True)


def graph_correl_features(dataset, df):
    """
    generates the graph of correlated features (heatmap matrix)
    :param dataset: dataset object
    :param df: data (as a dataframe)
    :return: None
    """
    # convert categorical to numerical
    for col in dataset.cat_cols:
        df[col].fillna('', inplace=True)
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # create correlation matrix with pandas
    corr = df.corr()

    # display heatmap
    plt.style.use('dark_background')
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
    plt.savefig(get_dataset_folder(dataset.uid) + '/graphs/_correl.png', transparent=True)


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
    scores = df_search[df_search.model_level == level].score_eval.values

    if dataset.best_is_min:
        # positive scores (e.g loss or error: min is best)
        y_lim1 = best_models.score_eval.abs().min()
        y_lim2 = best_models.score_eval.abs().quantile(0.5)
        y_lim1 -= (y_lim2 - y_lim1) * 0.02
    else:
        # negative scores (e.g. auc: max is best)
        y_lim1 = best_models.score_eval.abs().quantile(0.5)
        y_lim2 = best_models.score_eval.abs().max()
        y_lim2 += (y_lim2 - y_lim1) * 0.02

    # we will generate a list of the best values values over time
    mins = []
    best = METRIC_NULL
    for x in scores:
        if x < best:
            best = x
            mins.append(x)
        else:
            mins.append(best)
    mins = np.abs(mins)

    plt.figure(figsize=(6, 6))
    plt.style.use('dark_background')

    plt.plot(list(range(len(mins))), mins)
    plt.title('best score over time (level=%d)' % level)
    plt.xlabel('total searches')
    plt.ylabel('score')
    plt.ylim(y_lim1, y_lim2)
    plt.savefig(get_dataset_folder(dataset.uid) + '/graphs/_history_%d.png' % level, transparent=True)

    # we will generate a list of the max values values per model

    plt.figure(figsize=(6, 6))
    for model in best_models.model.unique()[:5][::-1]:
        scores = np.sort(np.abs(df_search[df_search.model == model].score_eval.values))[::-1]
        plt.plot(list(range(len(scores))), scores, label=model)

    plt.title('best score for 5 best models (level=%d)' % level)
    plt.xlabel('searches')
    plt.ylabel('score')
    plt.ylim(y_lim1, y_lim2)
    plt.legend(loc=1)
    plt.savefig(get_dataset_folder(dataset.uid) + '/graphs/_models_%d.png' % level, transparent=True)


def graph_predict(dataset, uuid, y, y_pred, part='eval'):
    """
    generate a graph prediction versus actuals (regression) or a confusion matrix (classification)
    :param dataset: dataset object
    :param uuid: id of the round
    :param y: actual values
    :param y_pred: predicted values
    :param part: part of the dataset
    :return: None
    """
    if dataset.problem_type == 'regression':
        fig = plt.figure(figsize=(6, 6))
        plt.style.use('dark_background')

        # plot a graph prediction versus actuals
        plt.scatter(y, y_pred, alpha=0.2, s=2)
        mn = min(min(y), min(y_pred))
        mx = max(max(y), max(y_pred))
        plt.plot((mn, mx), (mn, mx), color='r', lw=0.5)
        plt.xlim(mn, mx)
        plt.ylim(mn, mx)
        plt.xlabel('actuals')
        plt.ylabel('predict')
        plt.title('predict versus actuals (%s set)' % part)
    else:
        fig = plt.figure(figsize=(8, 6))
        plt.style.use('dark_background')

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

    plt.savefig(get_dataset_folder(dataset.uid) + '/graphs/predict_%s_%s.png' % (part, uuid), transparent=True)


def graph_pred_histogram(dataset_uid, uuid, y, part='eval'):
    """
    generate the histograph of predictions
    :param dataset_uid: id of the dataset
    :param uuid: id of the round (model)
    :param y: prediction values
    :param part: set (eval / train set)
    :return: None
    """
    plt.figure(figsize=(6, 6))
    plt.style.use('dark_background')
    plt.hist(y, label='prediction', bins=100)
    plt.title('histogram of predictions (%s set)' % part)
    plt.xlabel('values')
    plt.ylabel('frequencies')
    plt.savefig(get_dataset_folder(dataset_uid) + '/graphs/hist_%s_%s.png' % (part, uuid), transparent=True)
