import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def evaluate_models(X, Y):
    """
    Porównuje modele: MARS, Linear Regression, Tree, Random Forest.
    Zwraca ich MSE na zbiorze testowym.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    results = {}

    # MARS
    mars = Earth()
    mars.fit(X_train, Y_train)
    Y_pred_mars = mars.predict(X_test)
    results['MARS'] = mean_squared_error(Y_test, Y_pred_mars)

    # Linear Regression
    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)
    Y_pred_lin = linreg.predict(X_test)
    results['Linear Regression'] = mean_squared_error(Y_test, Y_pred_lin)

    # Decision Tree
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, Y_train)
    Y_pred_tree = tree.predict(X_test)
    results['Decision Tree'] = mean_squared_error(Y_test, Y_pred_tree)

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = mean_squared_error(Y_test, Y_pred_rf)

    return results
