import numpy as np  # biblioteka numeryczna
from matplotlib import pyplot as plt  # biblioteka do wykresów
from sklearn.linear_model import LogisticRegression  # regresja logistyczna z sklearn

from task2.compute import fit_and_get_coefs, compute_mse  # importujemy nasze funkcje pomocnicze
from task2.generate_data import generate_data  # importujemy funkcję generowania danych


def task2_simulation(L=100, sample_sizes=None):
    """
    Funkcja przeprowadza symulację, aby zbadać wpływ rozmiaru próbki na MSE estymatorów współczynników regresji logistycznej.

    Parametry:
    - L: liczba powtórzeń symulacji dla każdego rozmiaru próbki (domyślnie 100)
    - sample_sizes: lista rozmiarów próbek do przetestowania (jeśli None, ustawia domyślną listę)

    Zwraca:
    - mse_vs_n_full: lista MSE dla pełnego modelu (wszystkie 5 zmiennych)
    - mse_vs_n_3vars: lista MSE dla modelu uproszczonego (3 zmienne)
    """

    if sample_sizes is None:
        # Domyślne rozmiary próbek, które będziemy analizować
        sample_sizes = [50, 60, 70, 80, 90, 100, 200, 300, 500, 1000]

    # Prawdziwe współczynniki (generowane dane) — intercept + beta1..beta5
    beta0_true = 0.5
    beta_true = np.ones(5)

    # Tworzymy wektor true_coefs_full: [beta0, beta1, beta2, beta3, beta4, beta5]
    true_coefs_full = np.concatenate([[beta0_true], beta_true])

    # Listy do przechowywania wyników MSE
    mse_vs_n_full = []  # dla pełnego modelu
    mse_vs_n_3vars = []  # dla modelu z 3 zmiennymi

    # Iterujemy po wszystkich rozmiarach próbek
    for n in sample_sizes:
        coefs_full_all = []  # przechowuje estymowane współczynniki (pełny model)
        coefs_3vars_all = []  # przechowuje estymowane współczynniki (model uproszczony)

        # Dla każdego rozmiaru próbki powtarzamy eksperyment L razy
        for _ in range(L):
            # Generujemy dane treningowe
            X_full, y = generate_data(n, beta0=beta0_true, beta=beta_true)

            # Dopasowujemy pełny model i zapisujemy współczynniki
            coefs_full = fit_and_get_coefs(X_full, y)
            coefs_full_all.append(coefs_full)

            # Dopasowujemy model uproszczony (tylko 3 zmienne) i zapisujemy współczynniki
            X_3 = X_full[:, :3]  # bierzemy tylko pierwsze 3 kolumny
            model_3 = LogisticRegression(penalty='l2', solver='newton-cg')
            model_3.fit(X_3, y)
            coefs_3vars = np.concatenate([model_3.intercept_, model_3.coef_.flatten()])
            coefs_3vars_all.append(coefs_3vars)

        # Konwertujemy listy na numpy arrays (wymiary: L x liczba współczynników)
        coefs_full_all = np.array(coefs_full_all)
        coefs_3vars_all = np.array(coefs_3vars_all)

        # Obliczamy MSE dla pełnego modelu
        mse_full = compute_mse(coefs_full_all, true_coefs_full)
        mse_vs_n_full.append(mse_full)

        # Obliczamy MSE dla modelu uproszczonego (prawdziwe współczynniki to tylko beta0 i beta1..beta3)
        true_coefs_3 = np.concatenate([[beta0_true], beta_true[:3]])
        mse_3vars = compute_mse(coefs_3vars_all, true_coefs_3)
        mse_vs_n_3vars.append(mse_3vars)

    # Rysujemy wykres MSE w zależności od rozmiaru próbki — pełny model
    plt.figure()
    plt.plot(sample_sizes, mse_vs_n_full, marker='o')
    plt.xlabel('n')
    plt.ylabel('MSE (full model: 5 variables)')
    plt.title('MSE vs. n (beta0 + beta1..beta5)')
    plt.show()

    # Rysujemy wykres MSE w zależności od rozmiaru próbki — model uproszczony
    plt.figure()
    plt.plot(sample_sizes, mse_vs_n_3vars, marker='o')
    plt.xlabel('n')
    plt.ylabel('MSE (model 3 variables)')
    plt.title('MSE vs. n (beta0 + beta1..beta3)')
    plt.show()

    # Zwracamy wyniki w postaci dwóch list MSE
    return mse_vs_n_full, mse_vs_n_3vars
