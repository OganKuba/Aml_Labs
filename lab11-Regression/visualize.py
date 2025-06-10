import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualise_results(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        g_true_callable,
        *,
        model_a,  # first fitted model
        label_a="Nadaraya–Watson",
        model_b,  # second fitted model
        label_b="Smoothing spline",
        savepath: str | Path | None = None,
        title: str = "Kernel vs. Spline regression",
):
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    # Dense grid for smooth curves
    x_grid = np.linspace(0, 4, 500)
    y_grid_true = g_true_callable(x_grid)
    y_grid_a = model_a.predict(x_grid)
    y_grid_b = model_b.predict(x_grid)

    # --- plotting -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # scatter data
    ax.scatter(x_train, y_train, s=15, alpha=0.4, label="train")
    ax.scatter(x_test, y_test, s=15, alpha=0.4, label="test")

    # curves
    ax.plot(x_grid, y_grid_true, lw=2, label="true g(x)")
    ax.plot(x_grid, y_grid_a, lw=2, ls="--", label=label_a)
    ax.plot(x_grid, y_grid_b, lw=2, ls=":", label=label_b)

    # cosmetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.margins(x=0)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {savepath}")

    plt.show()
