import numpy as np
from fontTools.unicodedata.OTTags import NEW_SCRIPT_TAGS
from scipy.stats import norm

def simulate_data(n, p_true=10, p_noise=10, link='logistic'):
    """
       Generuje dane binarne zgodnie z modelem logistycznym lub probitowym.

       Parametry:
       -----------
       n : int
           Liczba obserwacji.
       p_true : int
           Liczba zmiennych istotnych (domyślnie 10).
       p_noise : int
           Liczba zmiennych zakłócających (domyślnie 10).
       link : str
           'logistic' lub 'probit'.

       Zwraca:
       -------
       tuple:
           X (ndarray), y (ndarray), beta (ndarray)
       """
    p = p_true + p_noise
    X = np.random.normal(0,1,size=(n, p))
    beta = np.zeros(p)
    beta[:p_true]=1

    lin_pred = X @ beta

    if link == 'logistic':
        probs = 1 / ( 1 + np.exp(-lin_pred))
    elif link == 'probit':
        probs = norm.cdf(lin_pred)
    else:
        raise ValueError('Link must be logistic or probit.')

    y = np.random.binomial(1, probs)
    return X, y, beta