import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def cross_validate_lambda(X, y, penalty='l1', l1_ratio=None, lambdas=None, cv=5):

    param_grid = {'C': [1.0 / lam for lam in lambdas]}
    lr = LogisticRegression(
        penalty=penalty,
        solver='saga',
        l1_ratio=l1_ratio,
        max_iter=5000,
    )
    clf = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy')
    clf.fit(X, y)
    best_lambda = 1.0 / clf.best_params_['C']
    best_score = clf.best_score_
    return best_lambda, best_score