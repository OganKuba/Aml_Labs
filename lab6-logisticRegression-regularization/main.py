# task1.py

import numpy as np
import pandas as pd
from data_loader  import load_prostate_data
from model_trainer import fit_logistic_regression
from profiler import plot_profile
from cross_validation import cross_validate_lambda

def main():
    # === Parametry ===
    x_filepath = 'prostate_x.txt'   # Plik z danymi X
    y_filepath = 'prostate_y.txt'   # Plik z etykietami y
    lambdas = np.logspace(-3, 1, 10)  # Przykładowe wartości lambdy w log-skali

    # === 1. Wczytaj dane ===
    X, y = load_prostate_data(x_filepath, y_filepath)
    y = np.array(y)

    # === 2. Modele: lasso, ridge, elastic net ===
    penalties = ['l1', 'l2', 'elasticnet']
    l1_ratio = 0.5  # Używane tylko dla elasticnet

    for penalty in penalties:
        print(f"\n=== Model: {penalty.upper()} ===")
        if penalty == 'elasticnet':
            coefs, scaler = fit_logistic_regression(X, y, penalty=penalty, l1_ratio=l1_ratio, lambdas=lambdas)
        else:
            coefs, scaler = fit_logistic_regression(X, y, penalty=penalty, lambdas=lambdas)

        # === 3. Drukuj współczynniki ===
        for idx, lam in enumerate(lambdas):
            print(f"Lambda={lam:.4f} -> Coefs: {np.round(coefs[idx], 4)}")

        # === 4. Profile współczynników ===
        var_names = [f"Gene {i+1}" for i in range(X.shape[1])]
        plot_profile(lambdas, coefs, variable_names=var_names)

        # === 5. Cross-validation ===
        best_lambda, best_score = cross_validate_lambda(
            X, y, penalty=penalty, l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
            lambdas=lambdas, cv=5
        )
        print(f"Najlepsza lambda (CV): {best_lambda:.4f} (Dokładność: {best_score:.4f})")

if __name__ == "__main__":
    main()
