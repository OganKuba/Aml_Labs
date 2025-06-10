# tree_visualization.py
# Plik odpowiada za wizualizację drzewa i parametrów.

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

def plot_parameter_comparison(results_df):
    """
    Funkcja tworzy wykresy porównujące parametry drzewa decyzyjnego.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Analiza parametrów drzewa decyzyjnego', fontsize=16)

    # Liczba węzłów vs dokładność testowa
    ax1 = axes[0, 0]
    for param in results_df['parameter'].unique():
        data = results_df[results_df['parameter'] == param]
        ax1.scatter(data['nodes'], data['test_acc'], label=param, s=60)
    ax1.set_xlabel('Liczba węzłów')
    ax1.set_ylabel('Dokładność testowa')
    ax1.set_title('Węzły vs Dokładność testowa')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Głębokość vs dokładność testowa
    ax2 = axes[0, 1]
    for param in results_df['parameter'].unique():
        data = results_df[results_df['parameter'] == param]
        ax2.scatter(data['depth'], data['test_acc'], label=param, s=60)
    ax2.set_xlabel('Głębokość drzewa')
    ax2.set_ylabel('Dokładność testowa')
    ax2.set_title('Głębokość vs Dokładność testowa')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Przeciwdziałanie przeuczeniu (dokładność treningowa vs testowa)
    ax3 = axes[1, 0]
    for param in results_df['parameter'].unique():
        data = results_df[results_df['parameter'] == param]
        ax3.scatter(data['train_acc'], data['test_acc'], label=param, s=60)
    ax3.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.5)
    ax3.set_xlabel('Dokładność treningowa')
    ax3.set_ylabel('Dokładność testowa')
    ax3.set_title('Analiza przeuczenia')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Średnia liczba węzłów dla każdego parametru
    ax4 = axes[1, 1]
    param_means = results_df.groupby('parameter')['nodes'].mean()
    param_means.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_xlabel('Parametr')
    ax4.set_ylabel('Średnia liczba węzłów')
    ax4.set_title('Rozmiar drzewa vs Parametr')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def visualize_tree_structure(X_train, y_train, feature_names, target_names, max_depth=3):
    """
    Funkcja rysuje strukturę drzewa decyzyjnego.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(clf,
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=8)
    plt.title(f'Struktura drzewa decyzyjnego (max_depth={max_depth})', fontsize=16)
    plt.show()

    return clf

def print_tree_rules(clf, feature_names):
    """
    Funkcja drukuje reguły drzewa w formie tekstowej.
    """
    tree_rules = export_text(clf, feature_names=feature_names)
    print("\n=== REGUŁY DRZEWA ===")
    print(tree_rules)
