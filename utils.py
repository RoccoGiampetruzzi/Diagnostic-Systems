import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split_dataframe(df, lb = True):

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Input should be a pandas dataframe')

    mean = df.loc[:, [col for col in df.columns if '_0' in col]].reset_index(drop=True).to_numpy()
    std = df.loc[:, [col for col in df.columns if '_1' in col]].reset_index(drop=True).to_numpy()
    worst = df.loc[:, [col for col in df.columns if '_2' in col]].reset_index(drop=True).to_numpy()

    if lb:
        labels = df['malignant'].to_numpy()

        return mean, std, worst, labels

    return mean, std, worst


def columns_mapping(df):

    cols = [col.split('_')[0] for col in df.columns]
    cols = list(dict.fromkeys(cols))
    index_cols = {val: i for i, val in enumerate(cols)}
    
    return index_cols