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


def evaluate_model(y_pred, y_test, model_name, print_results=True):

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)  # Row --> True labels, Columns --> Predicted Labels 

    if print_results:
        print(f"\nAccuracy of {model_name}: {accuracy}")

        print(f"\nConfusion Matrix for {model_name}:")
        print(conf_matrix)

        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, y_pred))

    return accuracy, conf_matrix


def custom_gridsearch(model_class, X_train, y_train, X_val, y_val, **parameters):

    best_accuracy = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
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

        accuracy, _, precision, recall, f1 = C.score(X_val, y_val, verbose=0)

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


def compare_IREP_RIPPER(irep, ripper, X_train, y_train, X_test, y_test, n_iterations):
    
    accuracy_irep = []
    accuracy_ripper = []
    num_conditions_irep = []
    num_conditions_ripper = []
    
    for i in range(n_iterations):

        irep.fit(X_train, y_train)
        y_pred1 = irep.predict(X_test)
        accuracy_irep.append(np.mean(y_pred1 == y_test))
        num_conditions_irep.append(sum([len(rule) for rule in irep.ruleset_.rules])) 

        ripper.fit(X_train, y_train)
        y_pred2 = ripper.predict(X_test)
        accuracy_ripper.append(np.mean(y_pred2 == y_test))
        num_conditions_ripper.append(sum([len(rule) for rule in ripper.ruleset_.rules])) 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(range(1, n_iterations + 1), accuracy_irep, label="IREP", marker='o')
    ax1.plot(range(1, n_iterations + 1), accuracy_ripper, label="RIPPER", marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Iterations')
    ax1.legend()

    ax2.plot(range(1, n_iterations + 1), num_conditions_irep, label="IREP", marker='o')
    ax2.plot(range(1, n_iterations + 1), num_conditions_ripper, label="RIPPER", marker='o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Number of Conditions in Clauses')
    ax2.set_title('Number of Conditions in Clauses over Iterations')
    ax2.legend()

    plt.tight_layout()
    plt.show()

