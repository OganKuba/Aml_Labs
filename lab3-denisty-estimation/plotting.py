import matplotlib.pyplot as plt
import numpy as np
from density_estimation import estimate_density

def plot_density_comparison(x_grid, theoretical_fn, estimated_density):
    plt.figure()
    plt.plot(x_grid, theoretical_fn(x_grid), label='Theoretical Density')
    plt.plot(x_grid, estimated_density, label='Estimated Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Theoretical vs. Estimated Density')
    plt.show()

def plot_mse_vs_sample_size(sample_sizes, mse_list):
    plt.figure()
    plt.plot(sample_sizes, mse_list, marker='o')
    plt.xlabel('Sample Size')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Sample Size')
    plt.grid()
    plt.show()

def plot_kernel_comparison(sample, theoretical_fn, x_grid, kernels, bandwidths):
    plt.figure(figsize=(10,6))
    for kernel in kernels:
        for bandwidth in bandwidths:
            _, estimated_density = estimate_density(sample, kernel=kernel, bandwidth=bandwidth)
            plt.plot(x_grid, estimated_density, label=f'{kernel}, bw={bandwidth}')
    plt.plot(x_grid, theoretical_fn(x_grid), 'k--', label='Theoretical')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Kernel and Bandwidth Influence')
    plt.show()

def plot_density_comparison_methods(x_grid, theoretical_fn, density_method1, density_method2):
    plt.figure()
    plt.plot(x_grid, theoretical_fn(x_grid), 'k-', label='Theoretical')
    plt.plot(x_grid, density_method1, 'b--', label='Method 1')
    plt.plot(x_grid, density_method2, 'r-.', label='Method 2')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Density Comparison: Method 1 vs. Method 2')
    plt.legend()
    plt.grid()
    plt.show()
