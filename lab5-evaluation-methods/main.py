# main.py

from synthetic_dataset import generate_synthetic_binary_dataset
from model_evaluation import fit_models
from model_plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    threshold_accuracy_plot
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    random_state = 42
    param_settings = [
        {"n": 100, "b": 0.5, "k": 5},
        # ... other parameter sets ...
    ]

    alpha = 0.0

    for params in param_settings:
        print(f"\n=== Parameters: n={params['n']}, b={params['b']}, k={params['k']} ===")
        X, y = generate_synthetic_binary_dataset(alpha, params['b'], params['k'], params['n'], random_state=random_state)
        y = y.values.ravel()  # flatten y

        # === Logistic Regression Model ===
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X, y)
        y_scores = model.predict_proba(X)[:, 1]  # probability of class 1

        # === (4) Plot ROC and Precision-Recall ===
        plot_roc_curve(y, y_scores, model_name=f"LR_n{params['n']}_b{params['b']}_k{params['k']}")
        plot_precision_recall_curve(y, y_scores, model_name=f"LR_n{params['n']}_b{params['b']}_k{params['k']}")

        # === (5) Train/Test Split ===
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=random_state)
        model.fit(X_train, y_train)
        y_scores_test = model.predict_proba(X_test)[:, 1]

        threshold_accuracy_plot(y_test, y_scores_test, model_name=f"LR_n{params['n']}_b{params['b']}_k{params['k']}")

if __name__ == "__main__":
    main()
