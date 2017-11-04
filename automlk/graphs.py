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

try:
    from wordcloud import WordCloud
    import_wordcloud = True
except:
    import_wordcloud = False


TRANSPARENT = False
SNS_STYLE = "whitegrid"


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
    plt.figure(figsize=(7, 7))
    sns.set(style=SNS_STYLE)
    if is_categorical:
        df = pd.DataFrame(values)
        df.columns = ['y']
        encoder = LabelEncoder()
        df['y'] = encoder.fit_transform(df['y'])
        values = df['y'].values
        sns.distplot(values, kde=False)
        x_labels = encoder.inverse_transform(list(range(max(values) + 1)))
        plt.xticks(list(range(max(values) + 1)), x_labels, rotation=90)
    else:
        sns.distplot(values)
    plt.title('distribution of %s (%s set)' % (col, part))
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
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].map(str))

    # create correlation matrix with pandas
    corr = df.corr()

    # display heatmap
    sns.set(style=SNS_STYLE)
    if dataset.n_cols > 50:
        plt.figure(figsize=(10, 10))
    elif dataset.n_cols > 20:
        plt.figure(figsize=(8, 8))
    elif dataset.n_cols > 10:
        plt.figure(figsize=(7, 7))
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
        # scores = np.sort(np.abs(df_search[df_search.model_name == model_name].cv_max.values))[::-1]
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

    plt.savefig(get_dataset_folder(dataset.dataset_id) + '/graphs/predict_%s_%s.png' % (part, round_id),
                transparent=TRANSPARENT)


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


def __standard_range(x, pmin, pmax):
    """
    calculate standard range with percentiles from pmin to pmax, eg. 1% to 99%

    :param x: list of values or np.array of dim 1
    :param pmin: percentile min (from 0 to 100)
    :param pmax: percentile max (from 0 to 100)
    :return: tuple (range_min, range_max)
    """
    y = np.sort(np.nan_to_num(x, 0))
    l = len(y)
    p1 = int(l * pmin / 100.)
    p2 = int(l * pmax / 100.) - 1
    return y[p1], y[p2]


def graph_regression_numerical(dataset_id, df, col, target):
    """
    display a reg scatter plot graph of col in x axis and target in y axis

    :param dataset_id: id of the dataset
    :param df: dataframe, with col and target values
    :param col: name of column
    :param target: name of target column
    :return:
    """
    sns.set(style=SNS_STYLE)
    g = sns.jointplot(x=col, y=target, data=df, kind="kde", size=7)
    g.plot_joint(plt.scatter, s=5, alpha=0.7)
    g.ax_joint.collections[0].set_alpha(0)
    plt.xlim(__standard_range(df[col].values, 1, 99))
    plt.ylim(__standard_range(df[target].values, 1, 99))
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/_col_%s.png' % col, transparent=TRANSPARENT)


def graph_regression_categorical(dataset_id, df, col, target):
    """
    display a boxplot graph of col in x axis and target in y axis

    :param dataset_id: id of the dataset
    :param df: dataframe, with col and target values
    :param col: name of column
    :param target: name of target column
    :return:
    """
    sns.set(style=SNS_STYLE)
    encoder = LabelEncoder()
    x = encoder.fit_transform(df[col].values)
    x_labels = encoder.inverse_transform(list(range(max(x) + 1)))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.boxplot(x=col, y=target, data=df, ax=ax)
    plt.xticks(list(range(max(x) + 1)), x_labels, rotation=90)
    plt.ylim(__standard_range(df[target].values, 1, 99))
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/_col_%s.png' % col, transparent=TRANSPARENT)


def graph_classification_numerical(dataset_id, df, col, target):
    """
    display a horizontal boxplot graph of col in x axis and target in y axis

    :param dataset_id: id of the dataset
    :param df: dataframe, with col and target values
    :param col: name of column
    :param target: name of target column
    :return:
    """
    sns.set(style=SNS_STYLE)
    plt.figure(figsize=(8, 7))
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target].values)
    y_labels = encoder.inverse_transform(list(range(max(y) + 1)))
    sns.boxplot(x=col, y=target, data=df, orient='h')
    plt.xlim(__standard_range(df[col].values, 1, 99))
    plt.yticks(list(range(max(y) + 1)), y_labels)
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/_col_%s.png' % col, transparent=TRANSPARENT)


def graph_classification_categorical(dataset_id, df, col, target):
    """
    display a heatmap of col in x axis and target in y axis

    :param dataset_id: id of the dataset
    :param df: dataframe, with col and target values
    :param col: name of column
    :param target: name of target column
    :return:
    """
    sns.set(style=SNS_STYLE)
    df['count'] = 1
    plt.figure(figsize=(8, 7))
    # convert col and target in numerical
    encoder = LabelEncoder()
    x = encoder.fit_transform(df[col].values)
    x_labels = encoder.inverse_transform(list(range(max(x) + 1)))
    y = encoder.fit_transform(df[target].values)
    y_labels = encoder.inverse_transform(list(range(max(y) + 1)))
    data = pd.pivot_table(df[[col, target, 'count']], values='count', index=target, columns=col, aggfunc=np.sum)
    sns.heatmap(data=data, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True)
    plt.xticks([x + 0.5 for x in list(range(max(x) + 1))], x_labels, rotation=90)
    plt.yticks([x + 0.5 for x in list(range(max(y) + 1))], y_labels, rotation=0)
    plt.savefig(get_dataset_folder(dataset_id) + '/graphs/_col_%s.png' % col, transparent=TRANSPARENT)


def graph_text(dataset_id, df, col):
    """
    display a wordcloud of the data of column col

    :param dataset_id: id of the dataset
    :param df: dataframe, with col values
    :param col: name of column
    :return:
    """
    if not import_wordcloud:
        return
    txt = " ".join([str(x) for x in df[col].values])
    wc = WordCloud(background_color="white", max_words=200, width=800, height=800)

    # generate word cloud
    wc.generate(txt)
    wc.to_file(get_dataset_folder(dataset_id) + '/graphs/_col_%s.png' % col)
