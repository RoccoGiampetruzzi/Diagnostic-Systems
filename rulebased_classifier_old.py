import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

## TODO : implement a function for intepretability of the classifier

class RuleBasedClassifier_old:
    def __init__(self, size, shp, texture, homogeneity, decision_option="majority"):
        
        self.mapping = None
        self.y = None
        self.X0 = None
        self.X1 = None
        self.X2 = None

        self.mean_0 = None
        self.mean_1 = None
        self.threshold = None

        if not isinstance(size, list):
            raise ValueError("size must be a list containing the features to be used in the size condition")
        
        if not isinstance(shp, list):
            raise ValueError("shape must be a list containing the features to be used in the shape condition")
        
        if not isinstance(texture, list):
            raise ValueError("texture must be a list containing the features to be used in the texture condition")
        
        if not isinstance(homogeneity, list):
            raise ValueError("homogeneity must be a list containing the features to be used in the homogeneity condition")
        
        if decision_option not in ["majority", "single", "all"]:
            raise ValueError("decision_option must be either 'majority' or 'single' or 'all'")
        
        self.size = np.array(size)
        self.shp = np.array(shp)
        self.texture = np.array(texture)
        self.homogeneity = np.array(homogeneity)
        self.decision_option = decision_option


    def fit(self, X, y):
        
        df = pd.concat([X, y], axis=1)
        self.mapping = columns_mapping(df)
        X0, X1, X2, y = split_dataframe(df)

        self.y = y
        self.X0 = X0
        self.X1 = X1
        self.X2 = X2

        #ben = np.where(self.y == 0)

        self.mean_0 = np.mean(self.X0, axis=0)
        self.mean_1 = np.mean(self.X1, axis=0)

        self.threshold = (self.mean_0 + 2 * self.mean_1) # for every feature column we have a threshold


    def hard_predict(self, X):

        X0, _, _= split_dataframe(X, lb=False)
        
        val = X0

        mask = (val < self.threshold).astype(int)

        #### Rule-based classification

        rhs = np.array([1, 1, 1, 1])  # Default case when decision_option is 'single'

        if self.decision_option == "majority":
            rhs = np.ceil(np.array([
                len(self.size) / 2, 
                len(self.shp) / 2, 
                len(self.texture) / 2, 
                len(self.homogeneity) / 2
            ])).astype(int)
        elif self.decision_option == "all":
            rhs = np.array([
                len(self.size), 
                len(self.shp), 
                len(self.texture), 
                len(self.homogeneity)
            ])

        # Condition 1: the conditions in self.size are not satisfied
        if len(self.size) == 0:
            cond1 = np.zeros(mask.shape[0], dtype=bool)
        else:
            cond1 = (mask[:, [self.mapping[x] for x in self.size]] == 0).sum(axis=1) >= rhs[0]

        # Condition 2: the conditions in self.shp are not satisfied
        if len(self.shp) == 0:
            cond2 = np.zeros(mask.shape[0], dtype=bool)
        else:
            cond2 = (mask[:, [self.mapping[x] for x in self.shp]] == 0).sum(axis=1) >= rhs[1]

        # Condition 3: the conditions in self.texture are not satisfied
        if len(self.texture) == 0:
            cond3 = np.zeros(mask.shape[0], dtype=bool)
        else:
            cond3 = (mask[:, [self.mapping[x] for x in self.texture]] == 0).sum(axis=1) >= rhs[2]

        # Condition 4: the conditions in self.homogeneity are not satisfied
        if len(self.homogeneity) == 0:
            cond4 = np.zeros(mask.shape[0], dtype=bool)
        else:
            cond4 = (mask[:, [self.mapping[x] for x in self.homogeneity]] == 0).sum(axis=1) >= rhs[3]


        predictions = np.where(cond1 | cond2 | cond3 | cond4, 1, 0)

        return predictions
    

    def score(self, X, y):

        y_pred = self.hard_predict(X)
        n_labels = int(np.max(y))+1
        confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
        
        # Counting entries of Confusion Matrix
        for true_label, pred_label in zip(y, y_pred):
            confusion_matrix[int(true_label), int(pred_label)] += 1 # Row --> True labels, Columns --> Predicted Labels
            
        # Computing accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        precision = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[0, 1])
        recall = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[1, 0])
        F1_score = 2*(precision*recall)/(precision + recall)

        return accuracy, precision, recall, F1_score, confusion_matrix
        
