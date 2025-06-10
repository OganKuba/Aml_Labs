import numpy as np
import pandas as pd


def generate_synthetic_binary_dataset(alpha, b, k, n, random_state=None):
    """
    Generate a synthetic binary classification dataset using logistic regression model.

    Parameters
    ----------
    alpha : float
        Intercept term in the logistic regression model.
    b : float
        Coefficient assigned to the first 5 features.
    k : int
        Number of additional noise features (features beyond the first 5).
    n : int
        Number of samples to generate.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x_df : pandas.DataFrame
        DataFrame containing feature matrix with columns ['x1', 'x2', ..., 'x(p)'].
    y_df : pandas.DataFrame
        DataFrame containing binary target variable in column ['target'].
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Total number of features (first 5 informative + k noise features)
    p = 5 + k

    # Initialize coefficient vector: first 5 features get coefficient b, rest get 0
    beta = np.zeros(p)
    beta[:5] = b

    # Generate feature matrix from standard normal distribution
    X = np.random.randn(n, p)

    # Compute the linear predictor: alpha + X * beta
    linear_predictor = alpha + np.dot(X, beta)

    # Apply the logistic function to get probabilities
    pi = 1 / (1 + np.exp(-linear_predictor))

    # Sample binary outcomes from Bernoulli(pi)
    y = np.random.binomial(1, pi, size=n)

    # Wrap the feature matrix and target in DataFrames
    columns = [f"x{i + 1}" for i in range(p)]
    x_df = pd.DataFrame(data=X, columns=columns)
    y_df = pd.DataFrame(data=y, columns=['target'])

    return x_df, y_df
