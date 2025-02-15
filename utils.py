import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools
from tqdm import tqdm


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



def evaluate_model(y_pred, y_test, model_name):

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy of {model_name}: {accuracy}")

    print(f"\nConfusion Matrix for {model_name}:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))


def custom_gridsearch(model_class, X_train, y_train, X_val, y_val, **parameters):

    best_accuracy = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_cm = None
    best_model = None
    optimal_combination = None

    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)
    
    pbar = tqdm(total=n_combinations, desc='Processing')

    for combination in combinations:

        params = dict(zip(param_names, combination))

        C = model_class(*params[param_names[0]], **{k: v for k, v in params.items() if k != param_names[0]})
        C.fit(X_train, y_train)

        accuracy, precision, recall, f1, _ = C.score(X_val, y_val)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_model = C
            optimal_combination = combination

        pbar.update(1)
    pbar.close()

    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best F1: {best_f1}")
    print(f"Best Precision: {best_precision}")
    print(f"Best Recall: {best_recall}")
    print(f"Best Combination: {optimal_combination}")

    return best_model, optimal_combination

