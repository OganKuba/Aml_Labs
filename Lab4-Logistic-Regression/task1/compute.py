import numpy as np  # Importujemy bibliotekę numpy do obliczeń numerycznych
from sklearn.linear_model import \
    LogisticRegression  # Importujemy klasyfikator regresji logistycznej z biblioteki sklearn


def compute_log_likelihood(y_true, p_pred):
    """
    Funkcja obliczająca logarytm wiarygodności (log-likelihood) dla modelu regresji logistycznej.
    Argumenty:
    - y_true: wektor etykiet rzeczywistych (0 lub 1)
    - p_pred: wektor prawdopodobieństw przewidywanych przez model
    """
    y_true = np.array(y_true, dtype=float)  # Konwertujemy etykiety na numpy array (w razie czego)

    eps = 1e-15  # Mała wartość zabezpieczająca przed log(0)
    p_pred = np.clip(p_pred, eps, 1 - eps)  # Upewniamy się, że wartości są w przedziale (eps, 1 - eps)

    # Obliczamy logarytm wiarygodności:
    # loglik = suma po wszystkich obserwacjach:
    # y_true * log(p_pred) + (1 - y_true) * log(1 - p_pred)
    loglik = np.sum(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))

    return loglik  # Zwracamy wartość log-likelihood


def fit_logistic_model(X, y, regularization=True):
    """
    Funkcja dopasowująca model regresji logistycznej do danych.
    Argumenty:
    - X: macierz cech (obserwacje w wierszach, cechy w kolumnach)
    - y: wektor etykiet (0 lub 1)
    - regularization: jeśli True, stosujemy regularizację L2 (domyślnie)
                      jeśli False, wyłączamy regularizację
    """
    if regularization:
        # Model z regularizacją L2 (domyślnie w LogisticRegression)
        model = LogisticRegression(penalty='l2', solver='newton-cg')
    else:
        # Model bez regularizacji (penalty=None)
        model = LogisticRegression(penalty=None, solver='newton-cg')

    # Dopasowujemy model do danych
    model.fit(X, y)

    # Wyciągamy prawdopodobieństwa przewidywane przez model (dla klasy 1)
    p_hat = model.predict_proba(X)[:, 1]

    # Zwracamy dopasowany model i przewidywane prawdopodobieństwa
    return model, p_hat
