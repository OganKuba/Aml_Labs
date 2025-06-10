import numpy as np

def generate_data(n, beta0=0.5, beta=np.ones(5)):
    """
    Funkcja generuje dane dla modelu regresji logistycznej.
    Argumenty:
    - n: liczba obserwacji do wygenerowania
    - beta0: wyraz wolny (intercept) w modelu logistycznym
    - beta: wektor współczynników regresji (domyślnie wektor jedynek długości 5)
    """
    p = len(beta)  # liczba cech

    # Generujemy macierz X: n wierszy, p kolumn — losowe wartości z rozkładu normalnego
    X = np.random.randn(n, p)

    # Obliczamy liniowy predyktor:
    # lin_pred = beta0 + X @ beta (mnożenie macierzy przez wektor)
    lin_pred = beta0 + np.dot(X, beta)

    # Obliczamy prawdopodobieństwa klasy 1 przy użyciu funkcji logistycznej (sigmoid):
    # p_i = 1 / (1 + exp(-lin_pred))
    p_i = 1.0 / (1.0 + np.exp(-lin_pred))

    # Generujemy etykiety y: dla każdej obserwacji losujemy 0 lub 1 z prawdopodobieństwem p_i
    y = np.random.binomial(1, p=p_i, size=n)

    # Zwracamy macierz cech X i wektor etykiet y
    return X, y

