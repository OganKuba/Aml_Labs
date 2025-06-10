from data_loader import load_prostate_data
from model_trainer import fit_logistic_regression
from profiler import plot_profile
from cross_validation import cross_validate_lambda


def main():
    # Load data
    X, y = load_prostate_data('prostate.csv')

    # Define lambdas
    lambdas = [0.01, 0.1, 1, 10, 100]

    # Lasso
    coefs_lasso, _ = fit_logistic_regression(X, y, penalty='l1', lambdas=lambdas)
    plot_profile(lambdas, coefs_lasso)
    best_lambda_lasso, score_lasso = cross_validate_lambda(X, y, penalty='l1', lambdas=lambdas)
    print(f"Lasso - Best lambda: {best_lambda_lasso}, CV score: {score_lasso}")

    # Ridge
    coefs_ridge, _ = fit_logistic_regression(X, y, penalty='l2', lambdas=lambdas)
    plot_profile(lambdas, coefs_ridge)
    best_lambda_ridge, score_ridge = cross_validate_lambda(X, y, penalty='l2', lambdas=lambdas)
    print(f"Ridge - Best lambda: {best_lambda_ridge}, CV score: {score_ridge}")

    # Elastic Net
    coefs_enet, _ = fit_logistic_regression(X, y, penalty='elasticnet', l1_ratio=0.5, lambdas=lambdas)
    plot_profile(lambdas, coefs_enet)
    best_lambda_enet, score_enet = cross_validate_lambda(X, y, penalty='elasticnet', l1_ratio=0.5, lambdas=lambdas)
    print(f"Elastic Net - Best lambda: {best_lambda_enet}, CV score: {score_enet}")


if __name__ == "__main__":
    main()
