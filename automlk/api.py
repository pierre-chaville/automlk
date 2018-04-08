from .dataset import make_dataset
from .worker import make_model
from .xyset import XySet
import pandas as pd


def fit_pipeline(j_dataset, j_model, df):
    """
    fit the model with data provided in the dataframe df, and information on the dataset in json format

    :param j_dataset: dataset info in json format
    :param j_model: model as a json
    :param df: data as dataframe (including x and y columns and columns not used)
    :return: trained pipeline
    """
    dataset = make_dataset(j_dataset)

    # create ds from df
    X = df[dataset.x_cols]
    y = df[dataset.y_col].values

    ds = XySet(X, y, X, y, pd.DataFrame(), [], pd.DataFrame(), [], None, None, None, None, None)

    model = make_model(dataset, j_model, ds)
    model.fit(X, y)

    return model


def predict_pipeline(j_dataset, model, df):
    """
    fit the model with data provided in the dataframe df, and information on the dataset in json format

    :param j_dataset: dataset info in json format
    :param model: trained pipeline
    :param df: data as dataframe (including x and y columns and columns not used)
    :return: prediction as a dataframe
    """
    dataset = make_dataset(j_dataset)

    # create ds from df
    X = df[dataset.x_cols]
    ds = XySet(X, [], X, [], pd.DataFrame(), [], pd.DataFrame(), [], None, None, None, None, None)
    return model.predict(X)
