import matplotlib.pyplot as plt

def plot_profile(lambdas, coefs, variable_names=None):
    plt.figure(figsize=(10, 6))
    num_vars = coefs.shape[1]
    for i in range(num_vars):
        var_label = variable_names[i] if variable_names is not None else f'Var {i + 1}'
        plt.plot(lambdas, coefs[:, i], label=var_label, alpha=0.5)

    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Regularization Path')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=5)
    plt.tight_layout()
    plt.show()