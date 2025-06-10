# data_loader.py
# Plik odpowiedzialny za wczytywanie i przygotowanie danych.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """
    Funkcja wczytuje dane o raku piersi i dzieli je na zbiory treningowy i testowy.
    Zwraca również nazwy cech i klasy.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    feature_names = data.feature_names
    target_names = data.target_names
    return X_train, X_test, y_train, y_test, feature_names, target_names
