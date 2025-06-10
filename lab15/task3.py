import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
TASK 3 – Gradient descent dla ważonej regresji logistycznej
===========================================================

∇R(θ) – WYPROWADZENIE (komentarz teoretyczny)
---------------------------------------------
R(θ) = -(1/n) Σ_i w_i [ y_i · (x_iᵀθ) + log(1 – σ(x_iᵀθ)) ] + 0.01‖θ‖²  
gdzie w_i = ‖x_i‖_∞,      σ(z)=1/(1+e^(–z))

d/dθ [y_i · (x_iᵀθ)]      = y_i x_i  
d/dθ log(1 – σ(z))        = –σ(z) x_i  
⇒ ∂/∂θ {…}               = (y_i – σ_i) x_i

∇R(θ) = (1/n) Σ_i w_i (σ_i – y_i) x_i+0.02 θ
------------------------------------------------
"""

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def weights_inf_norm(X):
    return np.max(np.abs(X), axis=1)

def risk(theta, X, y, w):
    z = X @ theta
    s = sigmoid(z)
    term = y * z + np.log(1-s)

    return -np.mean(w * term) + 0.01 * np.dot(theta, theta)

def gradient(theta, X, y, w):
    """
    ∇R(θ) = (1/n) Σ w_i (σ_i - y_i) x_i + 0.02 θ
    """
    n = y.size
    s = sigmoid(X @ theta)
    grad = (X.T @ ((w * (s - y)))) / n
    return grad + 0.02 * theta

def accuracy(theta, X, y):
    preds = sigmoid(X @ theta) > 0.5
    return np.mean(preds == y)

def load_or_generate():
    """
    Odczytuje train_data.csv i test_data.csv wygenerowane w Task 2.
    Jeśli plików brak, generuje je ponownie.
    """
    path_train = Path("train_data.csv")
    path_test  = Path("test_data.csv")

    if not (path_train.exists() and path_test.exists()):
        print("Plików z Task 2 nie znaleziono – generuję dane...")
        from task2 import generate_dataset           # funkcja z poprzedniego zadania
        cols = [f"X{i+1}" for i in range(10)]

        Xtr, ytr = generate_dataset(1000)
        pd.DataFrame(np.column_stack([Xtr, ytr]), columns=cols+["Y"]).to_csv(path_train, index=False)

        Xte, yte = generate_dataset(1000)
        pd.DataFrame(np.column_stack([Xte, yte]), columns=cols+["Y"]).to_csv(path_test,  index=False)

    train = pd.read_csv(path_train)
    test  = pd.read_csv(path_test)

    X_train, y_train = train.iloc[:, :-1].to_numpy(), train["Y"].to_numpy()
    X_test,  y_test  = test.iloc[:,  :-1].to_numpy(),  test["Y"].to_numpy()
    return X_train, y_train, X_test, y_test


# ---------- 3. Gradient Descent (GD) ----------

def run_gd(lr=0.01, epochs=100):
    X_tr, y_tr, X_te, y_te = load_or_generate()

    # dodajemy bias = 1 jako pierwszą kolumnę
    Xtr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
    Xte_b = np.hstack([np.ones((X_te.shape[0], 1)), X_te])

    w_tr = weights_inf_norm(X_tr)                 # w_i liczone bez biasu

    theta = np.zeros(Xtr_b.shape[1])              # inicjalizacja θ=0

    # metryki (epoch 0 = punkt startowy)
    risk_hist  = [risk(theta, Xtr_b, y_tr, w_tr)]
    acc_tr_hist = [accuracy(theta, Xtr_b, y_tr)]
    acc_te_hist = [accuracy(theta, Xte_b, y_te)]

    for _ in range(epochs):
        grad = gradient(theta, Xtr_b, y_tr, w_tr)
        theta -= lr * grad

        # zapisz statystyki po aktualizacji
        risk_hist.append(risk(theta, Xtr_b, y_tr, w_tr))
        acc_tr_hist.append(accuracy(theta, Xtr_b, y_tr))
        acc_te_hist.append(accuracy(theta, Xte_b, y_te))

    return risk_hist, acc_tr_hist, acc_te_hist


# ---------- 4. Uruchom & rysuj wykresy ----------

def main():
    risk_hist, acc_tr_hist, acc_te_hist = run_gd(lr=0.01, epochs=100)
    iters = np.arange(len(risk_hist))   # 0…100

    # --- wykres RISK.pdf ---
    plt.figure()
    plt.plot(iters, risk_hist, marker='o')
    plt.xlabel("Iteracja (epoch)")
    plt.ylabel("R(θ)")
    plt.title("Wartość ryzyka w kolejnych iteracjach")
    plt.grid()
    plt.savefig("RISK.pdf")
    plt.close()

    # --- wykres TRAIN.pdf ---
    plt.figure()
    plt.plot(iters, acc_tr_hist, marker='o')
    plt.xlabel("Iteracja (epoch)")
    plt.ylabel("Accuracy – zbiór treningowy")
    plt.title("Dokładność na treningu vs iteracja")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig("TRAIN.pdf")
    plt.close()

    # --- wykres TEST.pdf ---
    plt.figure()
    plt.plot(iters, acc_te_hist, marker='o')
    plt.xlabel("Iteracja (epoch)")
    plt.ylabel("Accuracy – zbiór testowy")
    plt.title("Dokładność na teście vs iteracja")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig("TEST.pdf")
    plt.close()

    print("Pliki RISK.pdf, TRAIN.pdf i TEST.pdf zostały zapisane.")

if __name__ == "__main__":
    main()







