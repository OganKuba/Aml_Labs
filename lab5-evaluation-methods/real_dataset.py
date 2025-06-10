from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_real_binary_dataset():
    """
    Load the breast cancer dataset from scikit-learn and
    return it as pandas DataFrames.

    Returns:
    --------
    x : pandas.DataFrame
        DataFrame containing the feature columns.
    y : pandas.DataFrame
        DataFrame containing the target column ('target'),
        where 0 = malignant and 1 = benign.
    """
    # Load the dataset from sklearn
    data = load_breast_cancer()

    # Convert the feature matrix to a DataFrame
    x = pd.DataFrame(data.data, columns=data.feature_names)

    # Convert the target array to a DataFrame with column name 'target'
    y = pd.DataFrame(data.target, columns=['target'])

    return x, y
