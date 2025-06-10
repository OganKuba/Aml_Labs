import numpy as np

def subset_accuracy(Y_true, Y_pred):
    """
    Returns the proportion of samples that have all labels correct.
    """
    return np.mean(np.all(Y_true == Y_pred, axis=1))

def hamming_score(Y_true, Y_pred):
    """
    Returns the average proportion of correctly predicted labels.
    """
    return np.mean(np.mean(Y_true == Y_pred, axis=1))