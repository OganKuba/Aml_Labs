import numpy as np
import matplotlib.pyplot as plt
# Usunięto błędny import: from sklearn.ensemble.tests.test_voting import X_scaled
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from simulator import simulate_data  # Zakładam, że ten plik jest w tym samym folderze


def fit_lasso(X, y):
    """
    Dopasowuje model Lasso (Regresja Logistyczna z karą L1) używając walidacji krzyżowej.
    """
    scaler = StandardScaler()
    # POPRAWKA: Użyj .fit_transform(), aby dopasować skaler i od razu przekształcić dane X.
    # Wynik zapisujemy w lokalnej zmiennej X_scaled.
    X_scaled = scaler.fit_transform(X)

    lasso = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        scoring='accuracy',
        max_iter=5000,
        cv=5,
        Cs=20,
        n_jobs=-1
    )
    # POPRAWKA: Teraz ta linia używa poprawnie przeskalowanych danych.
    lasso.fit(X_scaled, y)
    coefs = lasso.coef_.flatten()

    # Zwraca indeksy niezerowych współczynników (wybranych przez Lasso)
    selected = np.where(coefs != 0)[0]  # Zmieniono z > 0 na != 0, aby łapać też ujemne
    return selected


def compare_psr_fdr(selected, true_vars):
    """Oblicza Power of Selection Rate (PSR) i False Discovery Rate (FDR)."""
    t = set(true_vars)
    t_hat = set(selected)

    psr = len(t & t_hat) / len(t) if len(t) > 0 else 0
    fdr = len(t_hat - t) / len(t_hat) if len(t_hat) > 0 else 0

    return psr, fdr


def run_experiment(n, p_noise, link='logistic', L=100):
    """Uruchamia symulację L razy i oblicza średnie PSR i FDR."""
    psr_list = []
    fdr_list = []
    true_vars = list(range(10))  # Założenie, że pierwsze 10 zmiennych jest istotnych

    for _ in range(L):
        X, y, _ = simulate_data(n=n, p_true=10, p_noise=p_noise, link=link)
        selected = fit_lasso(X, y)
        psr, fdr = compare_psr_fdr(selected, true_vars)
        psr_list.append(psr)
        fdr_list.append(fdr)

    mean_psr = np.mean(psr_list)
    mean_fdr = np.mean(fdr_list)

    return mean_psr, mean_fdr


def plot_psr_fdr(x_values, psr_list, fdr_list, xlabel, title):
    """Rysuje wykresy PSR i FDR."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, psr_list, marker='o', label='PSR (Czułość)')
    plt.plot(x_values, fdr_list, marker='o', label='FDR (Odsetek fałszywych odkryć)')
    plt.xlabel(xlabel)
    plt.ylabel('Wartość')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Główny kod, który prawdopodobnie masz w task1.py ---
def run_sample_size_analysis():
    print("=== Analiza liczności próby ===")
    sample_sizes = [50, 100, 150, 200, 500]
    psr_results = []
    fdr_results = []

    for n in sample_sizes:
        print(f"Uruchamiam eksperyment dla n = {n}...")
        psr, fdr = run_experiment(n=n, p_noise=10)
        psr_results.append(psr)
        fdr_results.append(fdr)

    plot_psr_fdr(sample_sizes, psr_results, fdr_results,
                 'Liczność próby (n)',
                 'Wpływ liczności próby na PSR i FDR')


if __name__ == '__main__':
    run_sample_size_analysis()