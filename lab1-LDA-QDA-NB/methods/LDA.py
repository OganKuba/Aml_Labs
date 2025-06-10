import numpy as np
from scipy.special import expit

class LDA:
    def __init__(self):
        # Mean vector for class 0
        self.mu_0 = None
        #Mean vector for class 1
        self.mu_1 = None
        # Covariance matrix for both classes
        self.Sigma = None
        # Inverse of the covariance matrix
        self.Sigma_inv = None
        # Prior probability of class 0
        self.pi_0 = None
        # Prior probability of class 1
        self.pi_1 = None

    def fit(self, X, y):
        # Data for class 0
        X0 = X[y==0]
        X1 = X[y==1]

        # number of records
        n0 = X0.shape[0]
        n1 = X1.shape[0]
        n = float(n0 + n1)

        #Calculate mean
        self.mu_0 = np.mean(X0, axis=0)
        self.mu_1 = np.mean(X1, axis=0)

        # Covariance matrix calculation -> Sigma = # Sigma = (1 / (n - 1)) * (X - μ).T * (X - μ) for one clas
        # for two classes Sigma = ( (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1) ) / (n0 + n1 - 2)
        centered0 = X0 - self.mu_0
        centered1 = X1 - self.mu_1

        self.Sigma = (centered0.T @ centered0 + centered1.T @ centered1) / (n - 2)
        self.Sigma_inv = np.linalg.inv(self.Sigma)

        self.pi_0 = n0 / n
        self.pi_1 = n1 / n

    def predict_proba(self, Xtest):
        # Calculate mu_k * Sigma^-1
        mu0_Sigma_inv = self.mu_0 @ self.Sigma_inv
        mu1_Sigma_inv = self.mu_1 @ self.Sigma_inv

        # log(pi_1) - log(pi_0)
        log_diff = np.log(self.pi_1) - np.log(self.pi_0)

        # mean_diff = 0.5 * (mu1.T @ Sigma^-1 @ mu1 - mu0.T @ Sigma^-1 @ mu0)
        mean_diff = 0.5 * (self.mu_1 @ mu1_Sigma_inv - self.mu_0 @ mu0_Sigma_inv)

        # w = (mu1 - mu0).T @ Sigma^-1
        w = (self.mu_1 - self.mu_0) @ self.Sigma_inv

        result = log_diff - mean_diff + Xtest @ w

        prob_class1 = expit(result)

        return prob_class1

    def predict(self, Xtest):
        probability = self.predict_proba(Xtest)

        result = (probability > 0.5).astype(int)

        return result

    def get_params(self):
        return {
            'mu_0': self.mu_0,
            'mu_1': self.mu_1,
            'Sigma': self.Sigma,
            'Sigma_inv': self.Sigma_inv,
            'pi_0': self.pi_0,
            'pi_1': self.pi_1
        }




