import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.interpolate import UnivariateSpline

def read_data(filename):
    data = pd.read_csv(filename).dropna()

    x = data.iloc[:, 0].to_numpy(dtype=float).ravel()
    y = data.iloc[:, 1].to_numpy(dtype=float).ravel()

    idx = np.argsort(x)
    return x[idx], y[idx]

def split_data(x, y):
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    return [(train_idx, test_idx) for train_idx, test_idx in kf.split(x)]

def nw_kernel_regression(x_train, y_train, x_test, bandwidth):
    y_pred = []
    for x0 in x_test:
        weights = np.exp(-0.5 * ((x0 - x_train) / bandwidth) ** 2)
        s = weights.sum()
        if s == 0:
            y_pred.append(np.nan)
        else:
            y_pred.append(np.dot(weights, y_train) / s)
    return np.array(y_pred)

def nw_cross_validation(x, y, folds, bandwidths):
    mse = []
    for h in bandwidths:
        fold_mse = []
        for tr, te in folds:
            y_pred = nw_kernel_regression(x[tr], y[tr], x[te], h)
            fold_mse.append(np.nanmean((y[te] - y_pred) ** 2))
        mse.append(np.mean(fold_mse))
    return mse

def spline_cross_validation(x, y, folds, lambdas):
    mse = []
    for lam in lambdas:
        fold_mse = []
        for tr, te in folds:
            idx = np.argsort(x[tr])
            spline = UnivariateSpline(x[tr][idx], y[tr][idx], s=lam)
            fold_mse.append(np.mean((y[te] - spline(x[te])) ** 2))
        mse.append(np.mean(fold_mse))
    return mse

def plot_mse_vs_bandwidth(bandwidths, mse_nw):
    plt.figure()
    plt.plot(bandwidths, mse_nw, marker='o')
    plt.xlabel('Bandwidth (h)')
    plt.ylabel('Średni błąd kwadratowy (MSE)')
    plt.title('4-fold CV – Nadaraya-Watson')
    plt.grid(); plt.show()


def plot_mse_vs_lambda(lambdas, mse_spline):
    plt.figure()
    plt.plot(lambdas, mse_spline, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Średni błąd kwadratowy (MSE)')
    plt.title('4-fold CV – Smoothing splines')
    plt.grid(); plt.show()


def plot_fitted_curves(x, y, x_grid, y_nw, y_spline):
    plt.figure()
    plt.scatter(x, y, label='Dane', zorder=3)
    plt.plot(x_grid, y_nw, label='NW (h*)', linewidth=2)
    plt.plot(x_grid, y_spline, label='Spline (λ*)', linewidth=2)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Dopasowane krzywe')
    plt.legend(); plt.grid(); plt.show()

def main():
    x, y = read_data('weights.csv')
    folds = split_data(x, y)

    bandwidths = np.linspace(0.1, 2, 20)
    lambdas    = np.logspace(-2, 2, 20)

    mse_nw     = nw_cross_validation(x, y, folds, bandwidths)
    mse_spline = spline_cross_validation(x, y, folds, lambdas)

    plot_mse_vs_bandwidth(bandwidths, mse_nw)
    plot_mse_vs_lambda(lambdas, mse_spline)

    h_star      = bandwidths[np.nanargmin(mse_nw)]
    lambda_star = lambdas[np.argmin(mse_spline)]
    print(f'→ Najlepsze h*  = {h_star:.3f}')
    print(f'→ Najlepsze λ* = {lambda_star:.3g}')

    x_grid       = np.linspace(x.min(), x.max(), 200)
    y_nw_fit     = nw_kernel_regression(x, y, x_grid, h_star)
    spline_final = UnivariateSpline(x, y, s=lambda_star)
    y_spline_fit = spline_final(x_grid)

    plot_fitted_curves(x, y, x_grid, y_nw_fit, y_spline_fit)


if __name__ == '__main__':
    main()









