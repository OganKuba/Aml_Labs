import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd

def fit_models(X, y, random_state=None):
    """
    Fit Logistic Regression and Decision Tree models on the entire dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the fitted models.
    """
    logreg = LogisticRegression(max_iter=1000, random_state=random_state)
    tree = DecisionTreeClassifier(random_state=random_state)

    logreg.fit(X, y)
    tree.fit(X, y)

    return {"Logistic Regression": logreg, "Classification Tree": tree}

def estimate_error_refit(model, X, y):
    """
    Estimate error by refitting the model on the full dataset.

    Parameters
    ----------
    model : classifier object
        Fitted model.
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.

    Returns
    -------
    float
        Estimated error rate.
    """
    y_pred = model.predict(X)
    error = 1 - accuracy_score(y, y_pred)
    return error

def estimate_error_kfold(model, X, y, k=10, random_state=None):
    """
    Estimate error using k-fold cross-validation.

    Parameters
    ----------
    model : classifier object
        Model to be evaluated.
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.
    k : int, default=10
        Number of folds.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated error rate.
    """
    scores = cross_val_score(model, X, y, cv=k)
    error = 1 - np.mean(scores)
    return error

def estimate_error_bootstrap(model_class, X, y, B=100, random_state=None):
    """
    Estimate error using the .632 bootstrap method.

    Parameters
    ----------
    model_class : class
        Model class (not a fitted instance).
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.
    B : int, default=100
        Number of bootstrap replicates.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated error rate.
    """
    n = len(X)
    errors = []
    rng = np.random.default_rng(random_state)

    for b in range(B):
        # Bootstrap sample
        indices = rng.choice(n, size=n, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]

        # Out-of-bag sample
        mask_oob = ~np.isin(np.arange(n), indices)
        X_oob = X.iloc[mask_oob]
        y_oob = y.iloc[mask_oob]

        if len(y_oob) == 0:
            continue  # skip if no out-of-bag samples

        model = model_class()
        model.fit(X_boot, y_boot)

        y_pred = model.predict(X_oob)
        error = 1 - accuracy_score(y_oob, y_pred)
        errors.append(error)

    return np.mean(errors)

def estimate_error_bootstrap_632(model_class, X, y, B=100, random_state=None):
    """
    Estimate error using the .632 bootstrap correction.

    Parameters
    ----------
    model_class : class
        Model class (not a fitted instance).
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target variable.
    B : int, default=100
        Number of bootstrap replicates.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated error rate.
    """
    n = len(X)
    errors_oob = []
    rng = np.random.default_rng(random_state)

    for b in range(B):
        indices = rng.choice(n, size=n, replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]

        mask_oob = ~np.isin(np.arange(n), indices)
        X_oob = X.iloc[mask_oob]
        y_oob = y.iloc[mask_oob]

        if len(y_oob) == 0:
            continue  # skip if no out-of-bag samples

        model = model_class()
        model.fit(X_boot, y_boot)

        y_pred_oob = model.predict(X_oob)
        error_oob = 1 - accuracy_score(y_oob, y_pred_oob)
        errors_oob.append(error_oob)

    e_oob = np.mean(errors_oob)

    # Apparent error (fit and predict on full data)
    model_full = model_class()
    model_full.fit(X, y)
    y_pred_full = model_full.predict(X)
    e_app = 1 - accuracy_score(y, y_pred_full)

    # .632 correction
    error_632 = 0.368 * e_app + 0.632 * e_oob

    return error_632
