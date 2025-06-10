import numpy as np
from scipy.special import expit

class QDA:

    def __init__(self):
        self.mu_0 = None
        self.mu_1 = None
        self.Sigma_0 = None
        self.Sigma_1 = None
        self.Sigma_0_inv = None
        self.Sigma_1_inv = None
        self.pi_0 = None
        self.pi_1 = None

    def fit(self, X, y):
        # Divide the data into two classes
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Number of samples in each class
        n0 = X0.shape[0]
        n1 = X1.shape[0]
        n = float(n0 + n1)

        # Calculate the mean of each class
        self.mu_0 = np.mean(X0, axis=0)
        self.mu_1 = np.mean(X1, axis=0)

        # Calculate the covariance matrix of each class
        centered0 = X0 - self.mu_0
        centered1 = X1 - self.mu_1

        # Calculate the covariance matrix of each class
        self.Sigma_0 = (centered0.T @ centered0) / (n0 - 1)
        self.Sigma_1 = (centered1.T @ centered1) / (n1 - 1)

        # Calculate the inverse of the covariance matrix of each class
        self.Sigma_0_inv = np.linalg.inv(self.Sigma_0)
        self.Sigma_1_inv = np.linalg.inv(self.Sigma_1)

        # Calculate the prior probability of each class
        self.pi_0 = n0 / n
        self.pi_1 = n1 / n

    def predict_proba(self, Xtest):
        # Calculate the log of the ratio of the prior probabilities of the two classes
        log_pi_ratio = np.log(self.pi_1) - np.log(self.pi_0)

        # Calculate the half of the log determinant of the covariance matrix of each class
        half_log_det_0 = 0.5 * np.log(np.linalg.det(self.Sigma_0))
        half_log_det_1 = 0.5 * np.log(np.linalg.det(self.Sigma_1))

        # Calculate the constant term in the discriminant function
        const = log_pi_ratio + half_log_det_0 - half_log_det_1

        res = []
        # Calculate the discriminant function for each sample
        for x in Xtest:
            diff0 = x - self.mu_0
            diff1 = x - self.mu_1

            # Calculate the quadratic term in the discriminant function
            term0 = -0.5 * (diff1 @ self.Sigma_1_inv @ diff1)
            term1 = 0.5 * (diff0 @ self.Sigma_0_inv @ diff0)

            # Calculate the discriminant function
            result = const + term0 + term1
            res.append(result)

        res = np.array(res)

        prob_class1 = expit(res)

        return prob_class1

    def predict(self, Xtest):
        probability = self.predict_proba(Xtest)
        result = (probability > 0.5).astype(int)

        return result

    def get_params(self):
        return {
            'mu_0': self.mu_0,
            'mu_1': self.mu_1,
            'Sigma_0': self.Sigma_0,
            'Sigma_1': self.Sigma_1,
            'Sigma_0_inv': self.Sigma_0_inv,
            'Sigma_1_inv': self.Sigma_1_inv,
            'pi_0': self.pi_0,
            'pi_1': self.pi_1
        }
