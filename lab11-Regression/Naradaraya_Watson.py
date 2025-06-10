import numpy as np


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def nadaraya_watson_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_query: np.ndarray,
        *,
        bandwidth: float,
        kernel=gaussian_kernel,
        eps: float = 1e-12,
) -> np.ndarray:
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_query = np.asarray(x_query)

    u = (x_query[:, None] - x_train[None, :]) / bandwidth
    w = kernel(u)

    num = (w * y_train).sum(axis=1)
    den = w.sum(axis=1) + eps
    return num / den


class NadarayaWatsonRegressor:

    def __init__(self, bandwidth=0.2, kernel=gaussian_kernel):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._fitted = False

    def fit(self, X, y):
        self.x_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return nadaraya_watson_predict(
            self.x_train_,
            self.y_train_,
            np.asarray(X),
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        )
