# task1.py
# Plik główny uruchamiający analizy.

from data_loader import load_and_prepare_data
from tree_analysis import analyze_tree_parameters
from tree_visualization import plot_parameter_comparison, visualize_tree_structure, print_tree_rules
from pruning import cost_complexity_pruning
from ensemble_methods import compare_ensembles
from sklearn.tree import DecisionTreeClassifier

def main():
    print("ANALIZA DRZEWA DECYZYJNEGO (Raka piersi)")
    print("=" * 50)

    # 1. Wczytanie danych
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_data()
    print(f"Trening: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Cechy: {len(feature_names)}, Klasy: {target_names}")

    # 2. Analiza parametrów drzewa
    results_df = analyze_tree_parameters(X_train, X_test, y_train, y_test)
    plot_parameter_comparison(results_df)

    # 3. Wizualizacja struktury drzewa
    tree_clf = visualize_tree_structure(X_train, y_train, feature_names, target_names, max_depth=3)
    print_tree_rules(tree_clf, feature_names)

    # 4. Porównanie ensemble
    print("\n" + "=" * 50)
    print("PORÓWNANIE ENSEMBLE")
    ensemble_results = compare_ensembles(X_train, X_test, y_train, y_test)
    for model_name, acc in ensemble_results.items():
        print(f"{model_name}: Dokładność testowa = {acc:.3f}")

    # 5. Przycinanie drzewa
    print("\n" + "=" * 50)
    print("PRZYCINANIE DRZEWA")
    original_clf = DecisionTreeClassifier(random_state=42)
    original_clf.fit(X_train, y_train)
    pruned_clf, best_alpha = cost_complexity_pruning(X_train, X_test, y_train, y_test)

    # 6. Porównanie modeli przycinania
    print("\n=== PORÓWNANIE MODELI ===")
    print(f"Model bez przycinania: Treningowa={original_clf.score(X_train, y_train):.3f}, Testowa={original_clf.score(X_test, y_test):.3f}")
    print(f"Model po przycinaniu: Treningowa={pruned_clf.score(X_train, y_train):.3f}, Testowa={pruned_clf.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    main()
