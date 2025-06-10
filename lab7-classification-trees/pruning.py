# pruning.py
# Plik odpowiada za przycinanie drzewa decyzyjnego.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def cost_complexity_pruning(X_train, X_test, y_train, y_test):
    """
    Funkcja wykonuje przycinanie drzewa decyzyjnego przy użyciu metody cost-complexity pruning.
    """
    print("\n=== PRZYCINANIE DRZEWA (Cost-Complexity Pruning) ===")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs, train_scores, test_scores, node_counts = [], [], [], []

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        node_counts.append(clf.tree_.node_count)

    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    best_clf = clfs[best_idx]

    print(f"Najlepszy alpha: {best_alpha:.6f}")
    print(f"Najlepsza dokładność testowa: {test_scores[best_idx]:.3f}")

    plot_pruning_results(ccp_alphas, train_scores, test_scores, node_counts, best_alpha)

    return best_clf, best_alpha

def plot_pruning_results(ccp_alphas, train_scores, test_scores, node_counts, best_alpha):
    """
    Funkcja rysuje wykresy pokazujące wpływ parametru alpha na drzewo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Dokładność vs alpha
    ax1 = axes[0]
    ax1.plot(ccp_alphas, train_scores, marker='o', label='Treningowa', linewidth=2)
    ax1.plot(ccp_alphas, test_scores, marker='s', label='Testowa', linewidth=2)
    ax1.axvline(x=best_alpha, color='red', linestyle='--', label=f'Najlepszy α={best_alpha:.6f}')
    ax1.set_xlabel('Alpha (ccp_alpha)')
    ax1.set_ylabel('Dokładność')
    ax1.set_title('Dokładność vs Alpha')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Liczba węzłów vs alpha
    ax2 = axes[1]
    ax2.plot(ccp_alphas, node_counts, marker='d', color='green', linewidth=2)
    ax2.axvline(x=best_alpha, color='red', linestyle='--', label=f'Najlepszy α={best_alpha:.6f}')
    ax2.set_xlabel('Alpha (ccp_alpha)')
    ax2.set_ylabel('Liczba węzłów')
    ax2.set_title('Rozmiar drzewa vs Alpha')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.show()
