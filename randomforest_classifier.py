from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def best_randomforest(X,y,param_grid):

    model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("\nBest Hyperparameters selected are:", best_params)

    return best_model

