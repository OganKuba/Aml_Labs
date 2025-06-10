from density_estimation import generate_mixture_sample, theoretical_density, estimate_density
from mse_analysis import compute_mse, analyze_mse_vs_sample_size, analyze_kernel_and_bandwidth, compare_methods
from plotting import plot_density_comparison, plot_mse_vs_sample_size, plot_kernel_comparison
from classification import evaluate_all

def main():
    # Step 1: Generate Sample
    sample = generate_mixture_sample(200, seed=42)

    # Step 2: Theoretical Density
    theoretical_fn = theoretical_density()

    # Step 3: Kernel Density Estimate
    x_grid, kde_estimate = estimate_density(sample, kernel='gaussian', bandwidth=None)

    # Plot estimated vs. theoretical
    plot_density_comparison(x_grid, theoretical_fn, kde_estimate)

    # Step 4: Compute MSE
    mse = compute_mse(theoretical_fn, kde_estimate, x_grid)
    print(f"MSE: {mse:.5f}")

    # Step 5: Analyze MSE vs Sample Size
    analyze_mse_vs_sample_size()

    # Step 6: Analyze Kernel and Bandwidth Influence
    analyze_kernel_and_bandwidth()

    compare_methods()

    results = evaluate_all()
    print("\nClassification accuracies:")
    for model, acc in results.items():
        print(f"{model:12s}: {acc:.4f}")


if __name__ == "__main__":
    main()