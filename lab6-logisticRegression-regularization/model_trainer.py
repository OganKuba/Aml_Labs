# Importowanie potrzebnych bibliotek
import numpy as np  # Biblioteka do operacji numerycznych, głównie do pracy na tablicach
from sklearn.linear_model import LogisticRegression  # Model regresji logistycznej z biblioteki scikit-learn
from sklearn.preprocessing import StandardScaler  # Klasa do standaryzacji cech (skalowania)


def fit_logistic_regression(X, y, penalty='l1', l1_ratio=None, lambdas=None):
    """
    Funkcja dopasowuje model regresji logistycznej z regularyzacją dla różnych wartości lambda.

    Argumenty:
    X (array-like): Tablica z danymi wejściowymi (cechami).
    y (array-like): Tablica z etykietami docelowymi.
    penalty (str): Typ użytej kary (regularyzacji). Domyślnie 'l1' (Lasso).
                    Inne opcje to 'l2' (Ridge) lub 'elasticnet'.
    l1_ratio (float): Parametr mieszania dla kary 'elasticnet'.
    lambdas (list): Lista wartości parametru regularyzacji lambda do przetestowania.

    Zwraca:
    tuple: Krotka zawierająca (coefs, scaler)
           - coefs (np.array): Tablica NumPy z współczynnikami modelu dla każdej wartości lambda.
           - scaler (StandardScaler): Obiekt skalera dopasowany do danych X.
    """

    # Inicjalizacja obiektu StandardScaler, który przeskaluje dane tak,
    # aby miały średnią równą 0 i odchylenie standardowe równe 1.
    scaler = StandardScaler()

    # Dopasowanie skalera do danych X i jednoczesne ich przekształcenie (standaryzacja).
    # Standaryzacja jest ważna w modelach z regularyzacją.
    x_scaled = scaler.fit_transform(X)

    # Inicjalizacja pustej listy, która będzie przechowywać współczynniki modelu
    # dla każdej testowanej wartości lambda.
    coefs = []

    # Pętla iterująca po wszystkich podanych wartościach lambda.
    for lam in lambdas:
        # W scikit-learn parametr regularyzacji to C, który jest odwrotnością lambdy.
        # Mniejsza wartość C oznacza silniejszą regularyzację.
        C = 1.0 / lam

        # Inicjalizacja modelu regresji logistycznej z określonymi parametrami.
        lr = LogisticRegression(
            penalty=penalty,  # Ustawienie rodzaju regularyzacji (np. 'l1').
            l1_ratio=l1_ratio,  # Ustawienie proporcji dla kary 'elasticnet' (jeśli jest używana).
            C=C,  # Ustawienie siły regularyzacji.
            solver='saga',  # Wybór algorytmu optymalizacyjnego, 'saga' wspiera wszystkie rodzaje kar.
            max_iter=5000,  # Zwiększenie maksymalnej liczby iteracji, aby zapewnić zbieżność algorytmu.
        )

        # Trenowanie (dopasowanie) modelu na przeskalowanych danych.
        lr.fit(x_scaled, y)

        # Dodanie nauczonych współczynników do listy.
        # .coef_ zwraca tablicę 2D, więc .flatten() spłaszcza ją do 1D.
        coefs.append(lr.coef_.flatten())

    # Zwrócenie współczynników jako tablicy NumPy oraz obiektu skalera.
    # Skaler jest zwracany, aby można było zastosować tę samą transformację na nowych danych (np. teście).
    return np.array(coefs), scaler