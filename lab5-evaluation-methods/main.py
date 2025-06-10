# task1.py
from real_dataset import load_real_binary_dataset
from synthetic_dataset import generate_synthetic_binary_dataset
from model_evaluation import (
    fit_models,
    estimate_error_refit,
    estimate_error_kfold,
    estimate_error_bootstrap,
    estimate_error_bootstrap_632
)
from model_plots import (
    plot_roc_curve,
    plot_precision_recall_curve,
    threshold_accuracy_plot
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():
    random_state = 42
    alpha = 0.0
    b = 1.0
    k = 20
    n = 1000

    # 1(a) Load real dataset
    X_real, y_real = load_real_binary_dataset()
    y_real = y_real.values.ravel()

    print("=== Real Dataset ===")
    models_real = fit_models(X_real, y_real, random_state=random_state)
    for name, model in models_real.items():
        print(f"\nModel: {name}")
        print("Refitting error:", estimate_error_refit(model, X_real, y_real))
        print("10-fold CV error:", estimate_error_kfold(model, X_real, y_real, k=10, random_state=random_state))
        print("Bootstrap error:", estimate_error_bootstrap(model.__class__, X_real, y_real, random_state=random_state))
        print("Bootstrap .632 error:", estimate_error_bootstrap_632(model.__class__, X_real, y_real, random_state=random_state))

    # 1(b) Generate synthetic dataset
    X_syn, y_syn = generate_synthetic_binary_dataset(alpha, b, k, n, random_state=random_state)
    y_syn = y_syn.values.ravel()

    print("\n=== Synthetic Dataset ===")
    models_syn = fit_models(X_syn, y_syn, random_state=random_state)
    for name, model in models_syn.items():
        print(f"\nModel: {name}")
        print("Refitting error:", estimate_error_refit(model, X_syn, y_syn))
        print("10-fold CV error:", estimate_error_kfold(model, X_syn, y_syn, k=10, random_state=random_state))
        print("Bootstrap error:", estimate_error_bootstrap(model.__class__, X_syn, y_syn, random_state=random_state))
        print("Bootstrap .632 error:", estimate_error_bootstrap_632(model.__class__, X_syn, y_syn, random_state=random_state))

    # 4. ROC and Precision-Recall curves (synthetic data)
    param_settings = [
        {"n": 100, "b": 0.5, "k": 5},
        {"n": 1000, "b": 1.0, "k": 50}
    ]

    for params in param_settings:
        X, y = generate_synthetic_binary_dataset(alpha, params['b'], params['k'], params['n'], random_state=random_state)
        y = y.values.ravel()
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X, y)
        y_scores = model.predict_proba(X)[:, 1]
        model_name = f"LR_n{params['n']}_b{params['b']}_k{params['k']}"
        plot_roc_curve(y, y_scores, model_name=model_name)
        plot_precision_recall_curve(y, y_scores, model_name=model_name)

    # 5. Threshold analysis (synthetic data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_syn, y_syn, test_size=0.5, random_state=random_state
    )
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_scores_test = model.predict_proba(X_test)[:, 1]
    threshold_accuracy_plot(y_test, y_scores_test, model_name="LR_threshold_analysis")

if __name__ == "__main__":
    main()
