# data.py
import numpy as np
from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split

def generate_datasets():
    """
    Generuje dwa sztuczne zbiory danych: okręgi i klasyfikację.
    """
    X1, y1 = make_circles(n_samples=1000, noise=0.1, factor=0.5)
    X2, y2 = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0)
    return (X1, y1), (X2, y2)

def split_data(X, y, test_size=0.3, random_state=None):
    """
    Dzieli dane na zbiory treningowy i testowy.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
