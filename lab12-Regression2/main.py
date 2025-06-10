from data_generation import (
    generate_example1,
    generate_example2,
    generate_example3,
    generate_example4,
    generate_example5
)
from mars_analysis import run_mars, plot_X1_vs_Y, plot_Y_vs_Yhat
from real_data_comparison import evaluate_models
import seaborn as sns
import pandas as pd

def main():
    """
    Główny skrypt uruchamiający wszystkie zadania laboratoryjne.
    """
    examples = [
        ('Example 1', generate_example1),
        ('Example 2', generate_example2),
        ('Example 3', generate_example3),
        ('Example 4', generate_example4),
        ('Example 5', generate_example5)
    ]

    for name, generator in examples:
        X, Y = generator()
        mars_model = run_mars(X, Y)
        plot_X1_vs_Y(X, Y, mars_model, name)
        Y_pred = mars_model.predict(X)
        plot_Y_vs_Yhat(Y, Y_pred, name)

    # --- Część 2 ---
    # Dla uproszczenia, korzystamy np. z Boston dataset z sklearn (lub innego prawdziwego)
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    X_real = data.data
    Y_real = data.target

    results = evaluate_models(X_real, Y_real)
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Test MSE'])
    print("Porównanie modeli:")
    print(results_df)

if __name__ == "__main__":
    main()
