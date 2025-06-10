import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from pathlib import Path

from Naradaraya_Watson import NadarayaWatsonRegressor, nadaraya_watson_predict
from generate_data import generate_benchmark_data


def choose_bandwidth_cv(x, y, h_grid=np.linspace(0.05, 0.6, 12), n_folds=5):
    n = len(x)
    folds = np.array_split(np.arange(n), n_folds)
    best_h, best_mse = None, np.inf
    for h in h_grid:
        fold_mse = []
        for fold in folds:
            mask = np.zeros(n, bool)
            mask[fold] = True
            y_val_pred = nadaraya_watson_predict(x[~mask], y[~mask], x[mask], bandwidth=h)
            fold_mse.append(mean_squared_error(y[mask], y_val_pred))
        mse = np.mean(fold_mse)
        if mse < best_mse:
            best_h, best_mse = h, mse
    return best_h


def mse_vs_sample_size(
    n_values=(25, 50, 100, 200, 400, 800),
    noise_sigma: float = 0.1,
    seed: int = 0,
    savefig: str | None = "figures/mse_vs_n.png",
    show: bool = True,
):
    x_grid = np.linspace(0, 4, 1000)
    mse_nw, mse_sp = [], []

    for n in n_values:
        x_tr, y_tr, g_true = generate_benchmark_data(n, noise_sigma, seed + n)
        h_star = choose_bandwidth_cv(x_tr, y_tr)
        nw = NadarayaWatsonRegressor(h_star).fit(x_tr, y_tr)
        mse_nw.append(mean_squared_error(g_true(x_grid), nw.predict(x_grid)))

        spline = make_pipeline(
            SplineTransformer(degree=3, n_knots=12, include_bias=False),
            RidgeCV(alphas=np.logspace(-6, 2, 25)),
        ).fit(x_tr.reshape(-1, 1), y_tr)
        mse_sp.append(mean_squared_error(g_true(x_grid),
                                         spline.predict(x_grid.reshape(-1, 1))))

        print(f"n={n:4d} |  NW {mse_nw[-1]:.4e} |  Spline {mse_sp[-1]:.4e}")

    # plot
    plt.figure(figsize=(7, 4.3))
    plt.loglog(n_values, mse_nw, "o-", label="Nadaraya–Watson")
    plt.loglog(n_values, mse_sp, "s-", label="Smoothing spline")
    plt.grid(True, which="both", ls=":")
    plt.xlabel("sample size n"); plt.ylabel("MSE (dense grid)")
    plt.title("MSE vs. sample size (log–log)")
    plt.legend(); plt.tight_layout()

    if savefig:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefig, dpi=300)
        print("   >> zapisano:", savefig)

    if show:
        plt.show()
    else:
        plt.close()

