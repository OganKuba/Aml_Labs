import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # albo 'QtAgg', jeśli masz Qt
import matplotlib.pyplot as plt

from generate_data import generate_benchmark_data
from Naradaraya_Watson import NadarayaWatsonRegressor, nadaraya_watson_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV
from pathlib import Path
from mse import choose_bandwidth_cv

def compare_models(
    seed: int = 0,
    n_samples: int = 400,
    noise_sigma: float = 0.1,
    savefig: str | None = "figures/compare_nw_vs_spline.png",
    show: bool = True,
):
    # --- dane
    x, y, g_true = generate_benchmark_data(n_samples, noise_sigma, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(0.7 * n_samples)
    x_tr, y_tr = x[perm[:n_train]], y[perm[:n_train]]
    x_te, y_te = x[perm[n_train:]], y[perm[n_train:]]

    # Nadaraya–Watson
    h_star = choose_bandwidth_cv(x_tr, y_tr)
    nw = NadarayaWatsonRegressor(h_star).fit(x_tr, y_tr)
    mse_nw = mean_squared_error(y_te, nw.predict(x_te))

    # Smoothing spline
    spline = make_pipeline(
        SplineTransformer(degree=3, n_knots=12, include_bias=False),
        RidgeCV(alphas=np.logspace(-6, 2, 25)),
    ).fit(x_tr.reshape(-1, 1), y_tr)
    mse_sp = mean_squared_error(y_te, spline.predict(x_te.reshape(-1, 1)))

    print("=== Comparison (n =", n_samples, ") ===")
    print(f"  • Best h (NW)          = {h_star:.3f}")
    print(f"  • MSE test  – NW            = {mse_nw:.4f}")
    print(f"  • MSE test  – Spline        = {mse_sp:.4f}\n")

    # plot
    x_grid = np.linspace(0, 4, 500)
    plt.figure(figsize=(8, 4.5))
    plt.scatter(x_tr, y_tr, s=15, alpha=0.4, label="Train")
    plt.scatter(x_te, y_te, s=15, alpha=0.4, label="Test")
    plt.plot(x_grid, g_true(x_grid), lw=2, label="True g(x)")
    plt.plot(x_grid, nw.predict(x_grid), lw=2, ls="--",
             label=f"NW (h={h_star:.2f})")
    plt.plot(x_grid, spline.predict(x_grid.reshape(-1, 1)),
             lw=2, ls=":", label="Smoothing spline")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Nadaraya–Watson vs. Smoothing spline")
    plt.legend(); plt.tight_layout()

    if savefig:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefig, dpi=300)
        print("   >> saved:", savefig)

    if show:
        plt.show()
    else:
        plt.close()
