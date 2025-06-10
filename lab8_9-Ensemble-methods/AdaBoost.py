import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_X_y, check_array
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class AdaBoostBeta(BaseEstimator, ClassifierMixin):
    """
    Implementacja algorytmu AdaBoost z obliczaniem beta (modyfikowana wersja AdaBoost).
    """

    def __init__(self, n_estimators: int = 50,
                 base_estimator=None,
                 random_state: int | None = None):
        """
        Konstruktor klasy:
        - n_estimators: liczba iteracji (czyli liczba słabych klasyfikatorów)
        - base_estimator: bazowy klasyfikator, domyślnie: decision stump
        - random_state: generator liczb losowych dla powtarzalności wyników
        """
        self.n_estimators = n_estimators
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else DecisionTreeClassifier(max_depth=1, random_state=random_state)
        )
        self.random_state = random_state

    def fit(self, X, y):
        """
        Trenuje algorytm AdaBoost na danych (X, y).
        """
        # Sprawdzenie poprawności danych wejściowych
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        # Inicjalizacja wag próbek na równych wartościach
        w = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []  # lista słabych klasyfikatorów
        self.betas_ = []  # lista wag (beta) dla każdego klasyfikatora

        rng = np.random.RandomState(self.random_state)

        for m in range(self.n_estimators):
            # Tworzymy kopię bazowego klasyfikatora (aby nie nadpisywać)
            est = clone(self.base_estimator)
            if hasattr(est, "random_state"):
                est.set_params(random_state=rng.randint(0, np.iinfo(np.int32).max))

            # Trenujemy klasyfikator z uwzględnieniem wag
            est.fit(X, y, sample_weight=w)

            # Przewidujemy etykiety
            y_pred = est.predict(X)
            # Sprawdzamy, które próbki zostały źle sklasyfikowane
            miss = (y_pred != y)
            # Obliczamy błąd klasyfikatora (ważony)
            eps = np.dot(w, miss) / np.sum(w)

            # Jeżeli błąd jest zbyt duży (losowe klasyfikowanie), przerywamy boosting
            if eps >= 0.5:
                break;

            # Obliczamy wagę klasyfikatora (beta)
            beta = eps / (1 - eps + 1e-16)
            self.estimators_.append(est)
            self.betas_.append(beta)

            # Aktualizacja wag próbek:
            # - Dla poprawnie sklasyfikowanych wagi są mnożone przez beta
            correct = ~(miss)
            w[correct] *= beta

            # Normalizujemy wagi, aby sumowały się do 1
            w /= w.sum()

        self.betas_ = np.array(self.betas_)
        return self

    def score_matrix(self, X):
        """
        Tworzy macierz punktacji dla wszystkich klas.
        Każdy klasyfikator oddaje 'głos' proporcjonalny do log(1/beta).
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))

        # Przekształcamy beta do log(1/beta)
        log_inv_beta = np.log(1 / self.betas_)
        for est, weight in zip(self.estimators_, log_inv_beta):
            pred = est.predict(X)

            for idx, cls in enumerate(self.classes_):
                # Dodajemy wagę do odpowiedniej klasy
                scores[:, idx] += weight * (pred == cls)
        return scores

    def predict(self, X):
        """
        Zwraca etykiety klas dla nowych danych.
        """
        check_is_fitted(self)
        X = check_array(X)
        scores = self.score_matrix(X)
        # Zwracamy klasę z najwyższym wynikiem
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """
        Zwraca prawdopodobieństwo predykcji (normalizowane wyniki score_matrix).
        """
        check_is_fitted(self)
        X = check_array(X)
        scores = self.score_matrix(X)

        # Upewniamy się, że wyniki są nieujemne
        scores = np.maximum(scores, 0)
        norm = scores.sum(axis=1, keepdims=True) + 1e-16
        return scores / norm
