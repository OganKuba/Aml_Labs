from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_emotions_dataset(test_size=0.2, random_state=42):
    """
    Load the emotions dataset from OpenML, scale X, and convert
    label strings 'TRUE'/'FALSE' to integers 1/0.
    """
    X, Y = fetch_openml("emotions", version=4, return_X_y=True, as_frame=False)

    # 1. scale features
    X = StandardScaler().fit_transform(X)

    # 2. map label strings to integers
    #    (vectorised comparison is fastest)
    Y = (Y == b"TRUE").astype(np.int8)    # Y is a bytes/str array → 0 or 1

    # 3. train / test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, Y_train, Y_test