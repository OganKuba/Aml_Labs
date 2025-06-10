# experiments.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models import label_data, get_classifiers
from sklearn.svm import SVC


def run_experiment(X_train, y_train, X_test, y_test, g_values, n_repeats=10):
    """
    Uruchamia eksperyment dla różnych wartości g i powtarza go n_repeats razy.
    Zwraca słownik z wynikami.
    """
    base_estimator = SVC(probability=True, kernel='rbf')
    results = {method: {g: [] for g in g_values} for method in
               ["Naive", "SelfTraining", "LabelPropagation", "LabelSpreading"]}

    for g in tqdm(g_values, desc="Przetwarzanie eksperymentów"):
        for repeat in range(n_repeats):
            y_semi = label_data(y_train, g, random_state=repeat)
            classifiers = get_classifiers(base_estimator)
            for method_name, clf in classifiers.items():
                if method_name == "Naive":
                    # uczymy tylko na etykietowanych
                    mask = y_semi != -1
                    clf.fit(X_train[mask], y_semi[mask])
                else:
                    clf.fit(X_train, y_semi)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[method_name][g].append(acc)
    return results


def plot_results(results, filename="boxplot_results.png"):
    """
    Generuje boxploty wyników dla różnych wartości g.
    """
    methods = results.keys()
    g_values = sorted(next(iter(results.values())).keys())

    plt.figure(figsize=(12, 6))
    for idx, method in enumerate(methods):
        plt.subplot(1, len(methods), idx + 1)
        data = [results[method][g] for g in g_values]
        plt.boxplot(data, labels=g_values)
        plt.title(method)
        plt.xlabel("Liczba etykietowanych próbek na klasę (g)")
        plt.ylabel("Dokładność")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
