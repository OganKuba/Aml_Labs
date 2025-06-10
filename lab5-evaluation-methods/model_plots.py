# model_plots.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score
)

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} ROC curve")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_roc.png")
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"{model_name} Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_pr.png")
    plt.close()

def threshold_accuracy_plot(y_true, y_scores, model_name):
    thresholds = np.linspace(0, 1, 101)
    accuracies = []
    balanced_accuracies = []

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)

    plt.figure()
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy")
    plt.axvline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
    plt.axvline(np.mean(y_true), color='red', linestyle='--', label='Threshold = p(y=1)')
    plt.xlabel("Threshold t")
    plt.ylabel("Score")
    plt.title(f"Accuracy & Balanced Accuracy vs Threshold - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_threshold_plot.png")
    plt.close()
