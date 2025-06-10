# Importujemy potrzebne biblioteki
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from scipy.stats import chi2

# Importujemy własne implementacje AdaBoost i XGBoost
from AdaBoost import AdaBoostBeta
from XGB import XGB


def make_artificial(n_samples, rs=0):
    """
    Funkcja generująca sztuczny zbiór danych:
    - Losuje dane X z rozkładu normalnego (N(0,1))
    - Oblicza sumę kwadratów zmiennych dla każdej próbki
    - Etykieta y = 1 jeśli suma kwadratów przekracza wartość krytyczną rozkładu chi-kwadrat (df=10)
    - Inaczej etykieta = -1
    """
    rng = check_random_state(rs)
    X = rng.normal(size=(n_samples, 10))
    thr = chi2.ppf(0.5, df=10)  # wartość krytyczna rozkładu chi-kwadrat
    y = (np.sum(X ** 2, axis=1) > thr).astype(int)
    y[y == 0] = -1
    return X, y


def compute_errors(Xtr, Xte, ytr, yte, n_list):
    """
    Funkcja oblicza błąd klasyfikacji dla różnych modeli:
    - Drzewo decyzyjne (SingleTree)
    - Bagging
    - AdaBoost
    - RandomForest
    - XGBoost

    Dla każdego modelu testujemy różną liczbę estymatorów (n_list)
    i obliczamy błąd klasyfikacji (1 - accuracy).
    """
    # Słownik na błędy
    err = {m: [] for m in ["SingleTree", "Bagging",
                           "AdaBoost", "RandomForest", "XGBoost"]}

    # Trening pojedynczego drzewa decyzyjnego
    dt = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
    dt_err = 1 - accuracy_score(yte, dt.predict(Xte))
    err["SingleTree"] = [dt_err] * len(n_list)

    # Iteracja po różnych liczbach estymatorów
    for n in n_list:
        # Bagging
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=n, random_state=0, n_jobs=-1
        ).fit(Xtr, ytr)

        # AdaBoost (nasza implementacja)
        ada = AdaBoostBeta(n_estimators=n, random_state=0).fit(Xtr, ytr)

        # RandomForest
        rf = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1).fit(Xtr, ytr)

        # XGBoost (nasza implementacja)
        # Zamieniamy etykiety -1 na 0, bo XGBoost zwykle oczekuje etykiet w {0,1}
        yt_xgb = np.where(ytr == -1, 0, ytr)
        ye_xgb = np.where(yte == -1, 0, yte)
        xgb = XGB(n).fit(Xtr, yt_xgb)

        # Obliczamy błędy klasyfikacji
        err["XGBoost"].append(1 - accuracy_score(ye_xgb, xgb.predict(Xte)))
        err["Bagging"].append(1 - accuracy_score(yte, bag.predict(Xte)))
        err["AdaBoost"].append(1 - accuracy_score(yte, ada.predict(Xte)))
        err["RandomForest"].append(1 - accuracy_score(yte, rf.predict(Xte)))
    return err


def plot_curve(err, n_list, title):
    """
    Funkcja rysuje wykres błędów klasyfikacji w zależności od liczby estymatorów.
    """
    plt.figure()
    for m in err:
        plt.plot(n_list, err[m], marker="o", label=m)
    plt.xlabel("Number of Trees / Iterations")
    plt.ylabel("Test Error (1 − accuracy)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")


if __name__ == '__main__':
    # Lista liczności estymatorów, którą będziemy testować
    n_list = [1, 5, 10, 20, 40, 60, 80, 100]

    # Test na rzeczywistym zbiorze danych (rak piersi)
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    err_real = compute_errors(Xtr, Xte, ytr, yte, n_list)
    plot_curve(err_real, n_list, "Breast-Cancer (UCI)")

    # Test na sztucznym zbiorze danych (chi-kwadrat)
    Xa_tr, ya_tr = make_artificial(2000, rs=0)
    Xa_te, ya_te = make_artificial(10000, rs=1)
    err_art = compute_errors(Xa_tr, Xa_te, ya_tr, ya_te, n_list)
    plot_curve(err_art, n_list, "Syntetic dataset χ² (10 variables)")
