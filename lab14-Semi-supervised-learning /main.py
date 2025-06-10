# task1.py
from data import generate_datasets, split_data
from experiments import run_experiment, plot_results

def main():
    # Generowanie danych
    (X, y), _ = generate_datasets()  # np. weźmy zbiór okręgów

    # Podział danych
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    # Parametry
    g_values = [1, 2, 3, 4, 5]
    n_repeats = 10

    # Uruchomienie eksperymentów
    results = run_experiment(X_train, y_train, X_test, y_test, g_values, n_repeats)

    # Wyświetlenie wyników
    plot_results(results)

if __name__ == "__main__":
    main()
