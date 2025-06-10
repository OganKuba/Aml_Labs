import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

class MyBaggingClassifier:
    """
        Własna implementacja klasyfikatora Baggingowego
        (Bootstrap Aggregating) z drzewami decyzyjnymi jako bazowymi klasyfikatorami.
        """

    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []
        self.rng = np.random.default_rng(self.random_state)

    def fit(self, X, y):
        """
                Trenuje n_estimators bazowych klasyfikatorów na bootstrapowych próbkach.
        """
        n_samples = X.shape[0]
        self.estimators_ = []

        for i in range(self.n_estimators):
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            estimator = clone(self.base_estimator)
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

    def predict(self, X):
        """
               Predykcja na podstawie głosowania większościowego.
        """
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        maj_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return maj_vote

def compare_ensembles(X_train, X_test, y_train, y_test):
    """
    Porównuje trzy metody:
    - Pojedyncze drzewo
    - Bagging (własna implementacja)
    - Random Forest (gotowy)
    """
    results = {}

    # Pojedyncze drzewo
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train, y_train)
    y_pred_tree = single_tree.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    results['Single Tree'] = acc_tree

    # Bagging
    bagging = MyBaggingClassifier(n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    y_pred_bagging = bagging.predict(X_test)
    acc_bagging = accuracy_score(y_test, y_pred_bagging)
    results['Bagging'] = acc_bagging

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['Random Forest'] = acc_rf

    return results
