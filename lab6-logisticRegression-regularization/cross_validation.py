import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def cross_validate_lambda(X, y, penalty='l1', l1_ratio=None, lambdas=None, cv=5):
    """
    Dobiera optymalną wartość lambda dla regresji logistycznej z regularyzacją (L1 lub Elastic Net)
    za pomocą walidacji krzyżowej.

    Parametry:
    ----------
    X : array-like
        Macierz cech (dane wejściowe).
    y : array-like
        Wektor etykiet (dane wyjściowe).
    penalty : str, default='l1'
        Rodzaj regularyzacji ('l1' dla LASSO lub 'elasticnet' dla Elastic Net).
    l1_ratio : float, optional
        Parametr określający udział regularyzacji L1 w Elastic Net (od 0 do 1).
    lambdas : list or array-like
        Lista wartości parametru lambda do przetestowania.
    cv : int, default=5
        Liczba foldów w walidacji krzyżowej.

    Zwraca:
    -------
    best_lambda : float
        Najlepsza wybrana wartość lambda (regularyzacji).
    best_score : float
        Najlepszy wynik dokładności uzyskany podczas walidacji krzyżowej.
    """

    # Tworzymy siatkę parametrów (odwrotność lambdy, bo w sklearn używa się parametru C = 1/lambda)
    param_grid = {'C': [1.0 / lam for lam in lambdas]}

    # Konfigurujemy model regresji logistycznej:
    # - penalty określa typ regularyzacji ('l1' lub 'elasticnet')
    # - solver='saga' obsługuje oba typy regularyzacji
    # - l1_ratio (tylko dla elasticnet) określa udział L1 w regularyzacji mieszanej
    # - max_iter=5000 to maksymalna liczba iteracji do dopasowania modelu
    lr = LogisticRegression(
        penalty=penalty,
        solver='saga',
        l1_ratio=l1_ratio,
        max_iter=5000,
    )

    # GridSearchCV przeprowadza przeszukiwanie siatki parametrów z walidacją krzyżową
    clf = GridSearchCV(lr, param_grid, cv=cv, scoring='accuracy')

    # Dopasowujemy model na danych treningowych
    clf.fit(X, y)

    # Pobieramy najlepszą wartość parametru lambda (uwaga: C = 1/lambda)
    best_lambda = 1.0 / clf.best_params_['C']

    # Najlepszy wynik dokładności z walidacji krzyżowej
    best_score = clf.best_score_

    # Zwracamy najlepszą wartość lambda i wynik
    return best_lambda, best_score
