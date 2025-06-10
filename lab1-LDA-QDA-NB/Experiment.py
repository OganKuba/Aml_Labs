import pandas as pd
from sklearn.model_selection import train_test_split
from Generator import generate_data, accuracy_score
from LDA import LDA
from QDA import QDA
from NB import NB
import numpy as np


def run_experiments_scheme(scheme, vary='a', values=None, n_splits=25):
    results = {
        'LDA': {val: [] for val in values},
        'QDA': {val: [] for val in values},
        'NB': {val: [] for val in values}
    }

    for val in values:
        for _ in range(n_splits):
            # Generate data for this split
            if vary == 'a':
                X, y = generate_data(scheme, n=1000, a=val, rho=0.5)
            else:
                X, y = generate_data(scheme, n=1000, a=2.0, rho=val)

            # train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=np.random.randint(1e9)
            )

            # Fit and evaluate LDA
            lda = LDA()
            lda.fit(X_train, y_train)
            y_pred_lda = lda.predict(X_test)
            acc_lda = accuracy_score(y_test, y_pred_lda)
            results['LDA'][val].append(acc_lda)

            # Fit and evaluate QDA
            qda = QDA()
            qda.fit(X_train, y_train)
            y_pred_qda = qda.predict(X_test)
            acc_qda = accuracy_score(y_test, y_pred_qda)
            results['QDA'][val].append(acc_qda)

            # Fit and evaluate NB
            nb = NB()
            nb.fit(X_train, y_train)
            y_pred_nb = nb.predict(X_test)
            acc_nb = accuracy_score(y_test, y_pred_nb)
            results['NB'][val].append(acc_nb)

    return results
