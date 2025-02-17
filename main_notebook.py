#!/usr/bin/env python
# coding: utf-8

# # Diagnostic System

# In[21]:


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import *


# In[22]:


with open("wdbc.pkl", "rb") as f: 
    data = pickle.load(f)

print(type(data)) 
print(data.shape)
data


# In[23]:


data['malignant'].value_counts().plot(kind='bar')


# In[24]:


data.columns


# Column description:
# 
# | **Feature**           | **What It Measures**           | **Normal Cell Behavior**   | **Cancerous Cell Behavior**   |
# |----------------------|-----------------------------|--------------------------|-----------------------------|
# | **Radius**           | Size of the nucleus         | Small, uniform          | Larger, irregular         |
# | **Texture**         | Variation in brightness      | Smooth intensity        | High variability          |
# | **Perimeter**       | Length of boundary          | Shorter, round shape    | Longer, irregular shape   |
# | **Area**            | Total size                   | Smaller                 | Larger                    |
# | **Smoothness**      | Edge uniformity             | Smooth                  | Irregular edges           |
# | **Compactness**     | Shape density               | Dense, circular         | More stretched or jagged  |
# | **Concavity**       | Depth of inward curves      | Few or none             | Deep indentations         |
# | **Concave Points**  | Number of inward curves     | Few                     | Many                      |
# | **Symmetry**        | Balance of shape            | Symmetrical             | Asymmetrical              |
# | **Fractal Dimension** | Irregularity of edges     | Low                     | High                      |
# 
# 
# Next we create a test set that we will not touch during all the process but only at the end to test the models:

# In[25]:


data = data.drop(['id'], axis=1)
patient_X, patient_y = data.drop(['malignant'], axis=1), data['malignant']
patient_X, patient_X_test, patient_y, patient_y_test = train_test_split(patient_X, patient_y, test_size=0.1, random_state=42)

patient_df = pd.concat([patient_X, patient_y], axis=1)
patient_test = pd.concat([patient_X_test, patient_y_test], axis=1)


# In[26]:


patient_df[patient_df['malignant'] == 1].describe()


# In[27]:


patient_df[patient_df['malignant'] == 0].describe()


# Looking to the statistics of each feature of the dataframe, it is easy to notice the differences among the subset of the benign and of the malignant. The feature of the malignant subset have higher means, therefore we could exploit this in building our rule based classifier in the next section

# # CLASSIFICATION TASK

# ## Rule Based Classifier

# In[28]:


from rulebased_classifier import RuleBasedClassifier

size = ['area', 'perimeter', 'radius']
shape = ['smoothness', 'compactness', 'concavity', 'concave points']
texture = ['texture']
homog = [ 'area', 'smoothness']

C = RuleBasedClassifier(size, shape, texture, homog, decision_option="all")
C.fit(patient_X, patient_y)
C.classifier_rules()


# In[29]:


accuracy, confusion_matrix, precision, recall, F1_score = C.score(patient_X_test, patient_y_test, verbose=1)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", F1_score)


# The rule based classifier performs quiet well, but now we try to finetune it a little bit to try to improve its performances. In particular we cross validate and each time we remove a set of conditions in order to check whether some conditions are misleading for the model predictions

# In[30]:


patient_X_train, patient_X_val, patient_y_train, patient_y_val = train_test_split(patient_X, patient_y, test_size=0.2, random_state=42)

patient_df_train = pd.concat([patient_X_train, patient_y_train], axis=1)
patient_df_val = pd.concat([patient_X_val, patient_y_val], axis=1)


# In[31]:


constraints = [[[], shape, texture, homog] ,[size, [], texture, homog], 
               [size, shape, [], homog], [size, shape, texture, []], 
               [size, shape, texture, homog]]

options = ["all", "single", "majority"]


optimal_combination, best_model = custom_gridsearch(
    RuleBasedClassifier,  
    patient_X_train, patient_y_train, 
    patient_X_val, patient_y_val,
    constraints=constraints,
    decision_option=options
)


# In[32]:


C2 = RuleBasedClassifier(size, shape, [], homog, decision_option="all")
C2.fit(patient_X, patient_y)
accuracy, confusion_matrix, precision, recall, F1_score = C2.score(patient_X_test, patient_y_test, verbose=1)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", F1_score)


# From the search we discovered that the texture conditions do not help the model in discriminate between malignant and benign class, therefore by removing them we are able to achieve a better classification score on the test set.

# ## Random Forest Classifier

# We now adopt a more powerful approach: Random Forests. We expect this method to outperform the previous one, as the Random Forest classifier learns from the training data to identify the optimal rules for splitting the data and accurately classifying each sample. Moreover it is a larger model and we will perform k-fold cross validation to choose the best hyperparameters

# In[33]:


from randomforest_classifier import *

param = {
    'n_estimators': [25, 75, 150, 225, 300, 325],  
    'min_samples_split': [2, 5, 10], 
    'criterion' : ["gini", "entropy", "log_loss"]
}

randomforest_model = best_randomforest(patient_X, patient_y, param)

y_pred = randomforest_model.predict(patient_X_test)

accuracy, conf_matrix = evaluate_model(patient_y_test, y_pred, "Random Forest")


# The results are very good and the best model is the one using 300 decision trees and the entropy criterion. We can exploit the random forest also to understand which variables are more relevant and which are not, by looking to what features contributed the most in splitting the dataframe in the most meaningful way (minimizing the entropy).

# In[34]:


feature_importances = randomforest_model.feature_importances_
importance_df = pd.DataFrame({'Feature': patient_X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.gca().invert_yaxis() 
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Score")
plt.ylabel("Features")
plt.title("Most relevant features in Random Forest")
plt.show()


# ## Learned Rule Based Classifier

# The final approach that we propose is using a Rule Based Classifier that directly learns from the data what are the best possible literal and cluases that must be used according to some metric such as accuracy or entropy. 
# 
# The 2 models that we propose are IREP and its improved version RIPPER.

# In[35]:


import wittgenstein as lw


# In[50]:


irep_clf = lw.IREP() 
irep_clf.fit(patient_X_train, patient_y_train)

y_pred = irep_clf.predict(patient_X_test)
accuracy, conf_matrix = evaluate_model(y_pred, patient_y_test, 'IREP MODEL')


# In[98]:


irep_clf.out_model()


# In[108]:


len(irep_clf.ruleset_.rules[0])

sum([len(rule) for rule in irep_clf.ruleset_.rules])


# In[46]:


ripper_clf = lw.RIPPER() 
ripper_clf.fit(patient_X_train, patient_y_train)

y_pred = ripper_clf.predict(patient_X_test)
accuracy, conf_matrix = evaluate_model(y_pred, patient_y_test, 'RIPPER MODEL')


# In[47]:


ripper_clf.out_model()


# We compare the 2 models, and we discover that over multiple runs the performances are similar but the RIPPER algorithm tends to generate a much larger set of rules. It is not a good behaviour for us since reduces interpretability of the result conditions.

# In[113]:


compare_IREP_RIPPER(lw.IREP(), lw.RIPPER(), patient_X_train, patient_y_train, patient_X_val, patient_y_val, n_iterations=15)


# Both models perform very well, surpassing the performances of previous 2 approaches. Now we try to optimize further the IREP model with K-fold cross validation to determine the optimal hyperparameters.

# In[90]:


irep = lw.IREP(verbosity=0)

param_grid = {
    "prune_size": [0.1, 0.2, 0.33, 0.5],
    "n_discretize_bins": [5, 10, 15, 20],
    "max_total_conds": [5],
}

grid_irep = GridSearchCV(estimator=irep, param_grid=param_grid)
grid_irep.fit(patient_X, patient_y)


# In[94]:


best_params = grid_irep.best_params_
print("Best Parameters:", best_params)


# In[95]:


y_pred = grid_irep.predict(patient_X_test)
accuracy, conf_matrix = evaluate_model(y_pred, patient_y_test, 'IREP MODEL')


# In[96]:


grid_irep.best_estimator_.out_model()


# Optimizing the hyperparameters of the model, we obtain an impressive result since the model is able to almost perfectly discriminate among the samples of the 2 classes with only 1 error. Moreover it is able to determine a very simple and intuitive set of rules which enhance the interpretability of the model.
