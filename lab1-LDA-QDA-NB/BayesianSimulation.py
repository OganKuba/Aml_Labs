from Experiment import run_experiments_scheme
from Generator import generate_data
from Plot import boxplot_results
from methods import QDA
from methods import LDA

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def bayesian_simulation():
    a_values = [0.1, 0.5, 1, 2, 3, 5]
    rho_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

    # Run all experiments
    results_s1_a = run_experiments_scheme(scheme=1, vary='a', values=a_values, n_splits=20)
    results_s2_a = run_experiments_scheme(scheme=2, vary='a', values=a_values, n_splits=20)
    results_s1_rho = run_experiments_scheme(scheme=1, vary='rho', values=rho_values, n_splits=20)
    results_s2_rho = run_experiments_scheme(scheme=2, vary='rho', values=rho_values, n_splits=20)

    # Combine scheme=1 and scheme=2 (varying 'a') into a single PDF
    with PdfPages("BayesianSimulatedData1.pdf") as pdf:
        # Page 1: scheme=1
        plt.figure()
        boxplot_results(results_s1_a, parameter_name='a')  # no filename here
        pdf.savefig()
        plt.close()

        # Page 2: scheme=2
        plt.figure()
        boxplot_results(results_s2_a, parameter_name='a')
        pdf.savefig()
        plt.close()

    # Combine scheme=1 and scheme=2 (varying 'rho') into a single PDF
    with PdfPages("BayesianSimulatedData2.pdf") as pdf:
        # Page 1: scheme=1
        plt.figure()
        boxplot_results(results_s1_rho, parameter_name='rho')
        pdf.savefig()
        plt.close()

        # Page 2: scheme=2
        plt.figure()
        boxplot_results(results_s2_rho, parameter_name='rho')
        pdf.savefig()
        plt.close()

    # Generate data + LDA/QDA boundaries in a single figure
    a_chosen = 2
    rho_chosen = 0.5
    X, y = generate_data(scheme=2, a=a_chosen, rho=rho_chosen, n=1000, random_state=123)
    lda = LDA()
    lda.fit(X, y)
    qda = QDA()
    qda.fit(X, y)

    plt.figure()
    X0 = X[y == 0]
    X1 = X[y == 1]
    plt.scatter(X0[:, 0], X0[:, 1], marker='o', label="Class 0", alpha=0.6)
    plt.scatter(X1[:, 0], X1[:, 1], marker='x', label="Class 1", alpha=0.6)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # LDA boundary
    lda_scores = lda.predict_proba(grid_points) - 0.5
    lda_scores = lda_scores.reshape(xx.shape)

    # QDA boundary
    qda_scores = qda.predict_proba(grid_points) - 0.5
    qda_scores = qda_scores.reshape(xx.shape)

    plt.contour(xx, yy, lda_scores, levels=[0], linestyles=['-'], colors='red', linewidths=2)
    plt.contour(xx, yy, qda_scores, levels=[0], linestyles=['-'], colors='green', linewidths=2)

    plt.title(f"Scheme=2, a={a_chosen}, rho={rho_chosen}\nLDA (red) & QDA (green) boundaries")
    plt.legend()
    plt.savefig("BayesianSimulatedData3.pdf")
    plt.close()
