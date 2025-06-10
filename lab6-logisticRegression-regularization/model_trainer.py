import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fit_logistic_regression(X, y, penalty='l1', l1_ratio=None, lambdas=None):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    coefs = []

    for lam in lambdas:
        C = 1.0 / lam
        lr = LogisticRegression(
            penalty=penalty,
            l1_ratio=l1_ratio,
            C=C,
            solver='saga',
            max_iter=5000,
        )
        lr.fit(x_scaled, y)
        coefs.append(lr.coef_.flatten())

    return np.array(coefs), scaler