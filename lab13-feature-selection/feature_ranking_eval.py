from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import synthetic_data as sd

ImportanceMethod = Literal[
    "rf_mdi",
    "rf_perm",
    "boruta",
]


def _rf_classifier(random_state: int | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )


def rf_importance_mdi(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int | None = None,
) -> np.ndarray:
    rf = _rf_classifier(random_state)
    rf.fit(X, y)
    return rf.feature_importances_


def rf_importance_perm(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int | None = None,
        n_repeats: int = 10,
) -> np.ndarray:
    rf = _rf_classifier(random_state)
    rf.fit(X, y)
    pi = permutation_importance(
        rf, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=random_state
    )
    return pi.importances_mean


def boruta_importance(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int | None = None,
) -> np.ndarray:
    rf = _rf_classifier(random_state)
    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        random_state=random_state,
        verbose=0,
    )
    boruta.fit(X, y)

    score = np.full(X.shape[1], fill_value=0, dtype=float)
    score[boruta.support_] = 2
    score[boruta.support_weak_] = 1
    return score


_IMPORTANCE_FN: dict[ImportanceMethod, Callable[..., np.ndarray]] = {
    "rf_mdi": rf_importance_mdi,
    "rf_perm": rf_importance_perm,
    "boruta": boruta_importance,
}


def get_importance(
        X: np.ndarray,
        y: np.ndarray,
        method: ImportanceMethod,
        random_state: int | None = None,
) -> np.ndarray:
    return _IMPORTANCE_FN[method](X, y, random_state=random_state)


def perfect_ordering(
        scores: np.ndarray,
        k: int,
) -> bool:
    ranks_high_to_low = np.argsort(scores)[::-1]
    best_noise_rank = np.min(np.where(ranks_high_to_low >= k)[0])
    worst_signal_rank = np.max(np.where(ranks_high_to_low < k)[0])
    return worst_signal_rank < best_noise_rank


@dataclass
class SimulationResult:
    n: int
    p: int
    k: int
    L: int
    method: ImportanceMethod
    probability: float


def run_simulation(
        dataset: Literal["1", "2"],
        n: int,
        p: int,
        k: int,
        L: int,
        method: ImportanceMethod,
        *,
        random_state: int | None = 0,
) -> SimulationResult:
    rng = np.random.default_rng(random_state)
    successes = 0

    for _ in tqdm(range(L), desc=f"{method}-{dataset} (n={n},p={p},k={k})"):
        if dataset == "1":
            X, y = sd.make_dataset1(n, p, k, rng)
        else:
            X, y = sd.make_dataset2(n, p, k, rng)

        scores = get_importance(X, y, method, random_state=rng.integers(1e9))

        if perfect_ordering(scores, k):
            successes += 1

    prob = successes / L
    return SimulationResult(n=n, p=p, k=k, L=L, method=method, probability=prob)


def accuracy_vs_t(
        dataset: Literal["1", "2"],
        n: int,
        p: int,
        k: int,
        t_values: list[int],
        method: ImportanceMethod,
        *,
        test_size: float = 0.3,
        random_state: int | None = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    if dataset == "1":
        X, y = sd.make_dataset1(n, p, k, rng)
    else:
        X, y = sd.make_dataset2(n, p, k, rng)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rng.integers(1e9), stratify=y
    )

    full_scores = get_importance(X_train, y_train, method, random_state=rng.integers(1e9))
    ranked_idx = np.argsort(full_scores)[::-1]

    results = []
    for t in t_values:
        top_idx = ranked_idx[:t]
        clf = _rf_classifier(random_state=rng.integers(1e9))
        clf.fit(X_train[:, top_idx], y_train)
        y_pred = clf.predict(X_test[:, top_idx])
        acc = accuracy_score(y_test, y_pred)
        results.append((t, acc))

    df = pd.DataFrame(results, columns=["t", "accuracy"])
    return df


def plot_accuracy_curve(
        df: pd.DataFrame,
        outfile: str | Path,
        *,
        title: str | None = None,
) -> None:
    plt.figure()
    plt.plot(df["t"], df["accuracy"], marker="o")
    plt.xlabel("Number of top-ranked features (t)")
    plt.ylabel("Test accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(title or "Accuracy vs number of features")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


if __name__ == "__main__":
    from pathlib import Path

    param_grid = [
        {"n": 500, "p": 50, "k": 10},
        {"n": 200, "p": 50, "k": 5},
        {"n": 500, "p": 100, "k": 10},
    ]
    datasets = ["1", "2"]
    methods = ["rf_mdi", "rf_perm", "boruta"]

    L = 50  # number of simulation repeats
    RS = 42  # global seed for reproducibility
    t_values_master = [5, 10, 15, 20, 50, 100, 200, 300, 400, 500]

    out_root = Path("results")
    csv_dir = out_root / "csv"
    fig_dir = out_root / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sim_records = []
    for params in param_grid:
        for ds in datasets:
            for meth in methods:
                sim = run_simulation(
                    dataset=ds,
                    n=params["n"],
                    p=params["p"],
                    k=params["k"],
                    L=L,
                    method=meth,
                    random_state=RS,
                )
                sim_records.append(
                    {
                        "dataset": ds,
                        "method": meth,
                        **params,
                        "probability": sim.probability,
                    }
                )

    df_sim = pd.DataFrame(sim_records)
    df_sim.to_csv(csv_dir / "perfect_order_probabilities.csv", index=False)

    for (n_, p_, k_), grp in df_sim.groupby(["n", "p", "k"]):
        print(f"\n===  Perfect-order probability for n={n_}, p={p_}, k={k_}  (L={L}) ===")
        table = grp.pivot(index="dataset", columns="method", values="probability")
        table = table.reindex(columns=methods)  # preserve column order
        print(table.round(3).to_string())


    def run_boruta_validation(
            dataset: Literal["1", "2"],
            n: int,
            p: int,
            k: int,
            L: int,
            *,
            random_state: int | None = 0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        rows = []
        for rep in tqdm(range(L), desc=f"boruta-val-{dataset} (n={n},p={p},k={k})"):
            X, y = (
                sd.make_dataset1(n, p, k, rng)
                if dataset == "1"
                else sd.make_dataset2(n, p, k, rng)
            )

            rf = _rf_classifier(random_state=rng.integers(1e9))
            boruta = BorutaPy(
                rf, n_estimators="auto",
                random_state=rng.integers(1e9), verbose=0
            ).fit(X, y)

            support = boruta.support_
            tp = int(support[:k].sum())
            fp = int(support[k:].sum())
            fn = k - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / k

            rows.append(
                dict(dataset=dataset, n=n, p=p, k=k, rep=rep,
                     TP=tp, FP=fp, FN=fn,
                     precision=precision, recall=recall)
            )
        return pd.DataFrame(rows)


    boruta_frames = []
    for params in param_grid:
        for ds in datasets:
            boruta_frames.append(
                run_boruta_validation(
                    dataset=ds,
                    n=params["n"],
                    p=params["p"],
                    k=params["k"],
                    L=L,
                    random_state=RS,
                )
            )
    df_boruta = pd.concat(boruta_frames, ignore_index=True)

    summary = (df_boruta
               .groupby(["dataset", "n", "p", "k"])
               .agg(TP_mean=("TP", "mean"),
                    FP_mean=("FP", "mean"),
                    FN_mean=("FN", "mean"),
                    precision_mean=("precision", "mean"),
                    recall_mean=("recall", "mean"))
               .round(3))

    summary.to_csv(csv_dir / "boruta_validation_summary.csv")
    print("\n===  Boruta validation – mean over L repeats  ===")
    print(summary.to_string())

    for params in param_grid:
        for ds in datasets:
            for meth in methods:
                # clip t values so they do not exceed p
                t_vals = [t for t in t_values_master if t <= params["p"]]
                df_curve = accuracy_vs_t(
                    dataset=ds,
                    n=params["n"],
                    p=params["p"],
                    k=params["k"],
                    t_values=t_vals,
                    method=meth,
                    random_state=RS,
                )

                csv_name = (
                    f"accuracy_curve_ds{ds}_{meth}_n{params['n']}"
                    f"_p{params['p']}_k{params['k']}.csv"
                )
                df_curve.to_csv(csv_dir / csv_name, index=False)

                fig_name = csv_name.replace(".csv", ".png")
                title = (
                    f"{meth} – dataset {ds} "
                    f"(n={params['n']}, p={params['p']}, k={params['k']})"
                )
                plot_accuracy_curve(
                    df_curve,
                    outfile=fig_dir / fig_name,
                    title=title,
                )

    print(f"\nAll CSV files are in:  {csv_dir.resolve()}")
    print(f"All figures are in:   {fig_dir.resolve()}")
