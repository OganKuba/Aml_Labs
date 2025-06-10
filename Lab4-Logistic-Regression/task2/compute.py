import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_and_get_coefs(X, y):
    # Tworzymy model regresji logistycznej z regularyzacją L2
    model = LogisticRegression(penalty='l2', solver='newton-cg')

    # Dopasowujemy model do danych
    model.fit(X, y)

    # Pobieramy współczynniki modelu:
    # - model.intercept_ to wyraz wolny (bias)
    # - model.coef_ to macierz (1, n_features), więc spłaszczamy ją (flatten)
    coefs = np.concatenate([model.intercept_, model.coef_.flatten()])

    # Zwracamy wektor współczynników (bias + współczynniki regresji)
    return coefs


def compute_mse(est_coefs_matrix, true_coefs):
    # est_coefs_matrix to macierz współczynników (np. dla wielu powtórzeń symulacji)
    # true_coefs to wektor współczynników prawdziwych (lub zdefiniowanych w symulacji)

    # Odejmujemy wektor true_coefs od każdej kolumny est_coefs_matrix
    # Różnice (macierz błędów)
    diffs = est_coefs_matrix - true_coefs

    # Obliczamy sumy kwadratów różnic dla każdego wiersza (powtórzenia symulacji)
    sq_norms = np.sum(diffs ** 2, axis=1)

    # Obliczamy średnią kwadratową błędów (MSE)
    mse = np.mean(sq_norms)

    return mse
