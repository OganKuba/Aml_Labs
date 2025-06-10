import numpy as np


def generate_data(scheme, n=1000, a=2.0, rho=0.5, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    y = np.random.binomial(1, 0.5, size=n)

    X = np.zeros((n, 2))

    idx0 = (y == 0)
    idx1 = (y == 1)

    n0 = np.sum(idx0)
    n1 = np.sum(idx1)

    if scheme == 1:
        X[idx0, 0] = np.random.normal(0, 1, n0)
        X[idx0, 1] = np.random.normal(0, 1, n0)

        X[idx1, 0] = np.random.normal(a, 1, n1)
        X[idx1, 1] = np.random.normal(a, 1, n1)

    elif scheme == 2:
        cov_0 = np.array([[1, rho], [rho, 1]])
        mean_0 = [0, 0]
        X[idx0, :] = np.random.multivariate_normal(mean_0, cov_0, size=n0)

        cov_1 = np.array([[1, -rho], [-rho, 1]])
        mean_1 = [a, a]
        X[idx1, :] = np.random.multivariate_normal(mean_1, cov_1, size=n1)

    else:
        raise ValueError('Invalid scheme')

    return X, y


def accuracy_score(ytrue, ypred):
    return np.mean(ytrue == ypred)
