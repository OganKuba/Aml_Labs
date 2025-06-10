import numpy as np
import pandas as pd


def generate_example1(n=1000, p=50, noise_std=0.1):
    """
    Generuje dane dla Przykładu 1:
    X1 ~ U(0,4), X2,...,X50 ~ N(0,1)
    Y = sqrt(X1) + epsilon
    """

    X = np.zeros((n, p))
    X[:,0] = np.random.uniform(0, 4, n)
    X[:, 1:] = np.random.normal(0, 1, (n, p - 1))
    epsilon = np.random.normal(0, noise_std, n)

    Y = np.sqrt(X[:,0])+epsilon
    return pd.DataFrame(X, columns=[f'X{i + 1}' for i in range(p)]), Y

def generate_example2(n=1000, p=50, noise_std=0.1):
    """
    Przykład 2:
    Y = X1^2 + epsilon
    """
    X = np.zeros((n, p))
    X[:, 0] = np.random.uniform(0, 4, n)
    X[:, 1:] = np.random.normal(0, 1, (n, p - 1))
    epsilon = np.random.normal(0, noise_std, n)
    Y = X[:, 0]**2 + epsilon
    return pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)]), Y

def generate_example3(n=1000, p=50, noise_std=0.1):
    """
    Przykład 3:
    Y = (X1 - 0)+ + (X1 - 1)+ + epsilon
    """
    X = np.random.normal(0, 1, (n, p))
    epsilon = np.random.normal(0, noise_std, n)
    Y = np.maximum(X[:, 0] - 0, 0) + np.maximum(X[:, 0] - 1, 0) + epsilon
    return pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)]), Y

def generate_example4(n=1000, p=50, noise_std=0.1):
    """
    Przykład 4:
    Y = sin(X1) + epsilon
    """
    X = np.zeros((n, p))
    X[:, 0] = np.random.uniform(0, 4, n)
    X[:, 1:] = np.random.normal(0, 1, (n, p - 1))
    epsilon = np.random.normal(0, noise_std, n)
    Y = np.sin(X[:, 0]) + epsilon
    return pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)]), Y

def generate_example5(n=1000, p=50, noise_std=0.1):
    """
    Przykład 5:
    Y = I(X1 < 0)
    """
    X = np.random.normal(0, 1, (n, p))
    Y = (X[:, 0] < 0).astype(int)
    return pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)]), Y