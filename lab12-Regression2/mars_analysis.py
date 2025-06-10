import numpy as np
import matplotlib.pyplot as plt
from pyearth import Earth

def run_mars(X, Y):
    """
    Uruchamia model MARS na danych (X, Y).
    """
    model = Earth()
    model.fit(X, Y)
    return model

def plot_X1_vs_Y(X, Y, model, example_name):
    """
    Wykres: X1 vs Y (scatter) + predykcja MARS (linia).
    """
    x1_sorted = np.sort(X['X1'])
    y_pred_sorted = model.predict(X.assign(X1=x1_sorted))

    plt.figure()
    plt.scatter(X['X1'], Y, alpha=0.5, label='Prawdziwe Y')
    plt.plot(x1_sorted, y_pred_sorted, color='red', label='Predykcja MARS')
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.title(f'X1 vs Y ({example_name})')
    plt.legend()
    plt.show()

def plot_Y_vs_Yhat(Y, Y_pred, example_name):
    """
    Wykres: prawdziwe Y vs przewidywane Y (scatter).
    """
    plt.figure()
    plt.scatter(Y, Y_pred, alpha=0.5)
    plt.xlabel('Prawdziwe Y')
    plt.ylabel('Predykcja MARS')
    plt.title(f'Y vs Predykcja ({example_name})')
    plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')
    plt.show()
