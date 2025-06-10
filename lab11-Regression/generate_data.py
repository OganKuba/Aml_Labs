import numpy as np


def generate_benchmark_data(n_samples=200, noise_sigma=0.1, seed=None):
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, 4.0, size=n_samples)

    def g(x):
        return 4.26 * (np.exp(-x) - 4 * np.exp(-2 * x) + 3 * np.exp(-3 * x))

    noise = rng.normal(0.0, noise_sigma, size=n_samples)
    y = g(x) + noise

    return x, y, g
