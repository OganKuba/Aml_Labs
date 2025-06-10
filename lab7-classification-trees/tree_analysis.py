# tree_analysis.py
# Plik zawiera funkcje do analizowania i oceny modeli drzewa decyzyjnego.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def evaluate_tree_model(clf, X_train, X_test, y_train, y_test):
    """
    Funkcja trenuje model drzewa decyzyjnego i zwraca metryki takie jak dokładność,
    liczba węzłów i głębokość drzewa.
    """
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    n_nodes = clf.tree_.node_count
    depth = clf.tree_.max_depth
    return train_acc, test_acc, n_nodes, depth

def analyze_parameter_values(param_name, param_values, X_train, X_test, y_train, y_test):
    """
    Funkcja bada wpływ wybranego parametru drzewa na strukturę i dokładność modelu.
    """
    print(f"\n=== ANALIZA PARAMETRU: {param_name.upper()} ===")
    results = []

    for value in param_values:
        # Tworzymy model z odpowiednią wartością parametru
        kwargs = {param_name: value, 'random_state': 42}
        clf = DecisionTreeClassifier(**kwargs)

        # Oceniamy model
        train_acc, test_acc, n_nodes, depth = evaluate_tree_model(clf, X_train, X_test, y_train, y_test)

        # Zapisujemy wyniki
        results.append({
            'parameter': param_name,
            'value': str(value),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'nodes': n_nodes,
            'depth': depth
        })

        print(f"{param_name.capitalize()}={value}: Węzły={n_nodes}, Głębokość={depth}, Dokładność testowa={test_acc:.3f}")

    return results

def analyze_tree_parameters(X_train, X_test, y_train, y_test):
    """
    Funkcja wykonuje analizę kilku parametrów drzewa.
    """
    all_results = []
    parameter_configs = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 10, 20, 50]
    }

    for param_name, param_values in parameter_configs.items():
        results = analyze_parameter_values(param_name, param_values, X_train, X_test, y_train, y_test)
        all_results.extend(results)  # Poprawka: extend() zamiast append() bo chcemy 1 duży DataFrame

    return pd.DataFrame(all_results)
