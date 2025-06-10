from compare import compare_models
from mse import mse_vs_sample_size


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == "__main__":
    # Part 3-5 (Compare Nadaraya-Watson and Spline)
    compare_models(seed=0, n_samples=400, noise_sigma=0.1,
                   savefig="figures/compare_nw_vs_spline.png", show=True)

    # Part 6 (MSE vs. sample size)
    mse_vs_sample_size(
        n_values=[25, 50, 100, 200, 400, 800],
        noise_sigma=0.1,
        seed=0,
        savefig="figures/mse_vs_n.png",
        show=True,
    )
