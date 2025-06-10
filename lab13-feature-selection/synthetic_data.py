from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2


def _draw_features(n: int, p: int,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.standard_normal(size=(n, p))


def make_dataset1(n: int, p: int, k: int,
                  rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= p:
        raise ValueError("k must be in the range 1 to p")
    X = _draw_features(n, p, rng)
    threshold = chi2.ppf(0.5, df=k)
    y = (np.sum(X[:, :k] ** 2, axis=1) > threshold).astype(int)
    return X, y


def make_dataset2(n: int, p: int, k: int,
                  rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= p:
        raise ValueError("k must be in the range 1 to p")
    X = _draw_features(n, p, rng)
    y = (np.sum(np.abs(X[:, :k]), axis=1) > k).astype(int)
    return X, y


def generate_dataframe(dataset: str, n: int, p: int, k: int,
                       rng_seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    if dataset == "1":
        X, y = make_dataset1(n, p, k, rng)
    elif dataset == "2":
        X, y = make_dataset2(n, p, k, rng)
    else:
        raise ValueError("dataset must be '1' or '2'")

    df = pd.DataFrame(X, columns=[f"X{j + 1}" for j in range(p)])
    df["Y"] = y
    return df
