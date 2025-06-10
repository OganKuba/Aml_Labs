import numpy as np
from density_estimation import generate_mixture_sample, theoretical_density, estimate_density, \
    generate_artificial_sample
from plotting import plot_mse_vs_sample_size


# Funkcja obliczająca MSE między teoretyczną gęstością a oszacowaną
def compute_mse(theoretical_fn, estimated_density, x_grid):
    # Obliczamy prawdziwą gęstość w punktach x_grid
    true_density = theoretical_fn(x_grid)
    # MSE: średnia kwadratowa różnica między true_density i estimated_density
    mse = np.mean((true_density - estimated_density) ** 2)
    return mse  # zwracamy wartość MSE


# Funkcja analizująca jak MSE zmienia się wraz z rozmiarem próbki
def analyze_mse_vs_sample_size():
    # Lista różnych rozmiarów próbek, które chcemy przetestować
    sample_sizes = [50, 100, 200, 500, 1000]

    # Tworzymy gęstą siatkę punktów (10 000 punktów) do rysowania gęstości
    x_grid = np.linspace(2, 12, 10000)

    # Pobieramy funkcję teoretycznej gęstości (mieszanka Gaussa)
    theoretical_fn = theoretical_density()

    # Lista na wyniki MSE dla różnych rozmiarów próbek
    mse_list = []

    # Iterujemy po wszystkich rozmiarach próbek
    for n in sample_sizes:
        # Generujemy próbkę danych z rozkładu mieszanego
        sample = generate_mixture_sample(n, seed=42)

        # Szacujemy gęstość rozkładu dla tej próbki
        _, estimated_density = estimate_density(sample, x_grid=x_grid)

        # Obliczamy MSE między teoretyczną a estymowaną gęstością
        mse = compute_mse(theoretical_fn, estimated_density, x_grid)

        # Zapisujemy MSE do listy
        mse_list.append(mse)

    # Rysujemy wykres MSE w zależności od rozmiaru próbki
    plot_mse_vs_sample_size(sample_sizes, mse_list)


# Funkcja analizująca wpływ rodzaju jądra i szerokości pasma na estymację gęstości
def analyze_kernel_and_bandwidth():
    # Lista różnych jąder do przetestowania w KDE
    kernels = ['gaussian', 'tophat', 'epanechnikov']

    # Lista różnych szerokości pasma do przetestowania
    bandwidths = [0.3, 0.5, 1.0]

    # Generujemy jedną próbkę danych (rozmiar 200)
    sample = generate_mixture_sample(200, seed=42)

    # Pobieramy funkcję teoretycznej gęstości
    theoretical_fn = theoretical_density()

    # Tworzymy siatkę punktów (1000 punktów) w przedziale [2,12]
    x_grid = np.linspace(2, 12, 1000)

    # Importujemy funkcję do rysowania porównania jąder (z plotting.py)
    from plotting import plot_kernel_comparison

    # Wywołujemy funkcję, która rysuje porównanie gęstości dla różnych jąder i bandwidthów
    plot_kernel_comparison(sample, theoretical_fn, x_grid, kernels, bandwidths)

def compare_methods(sample_size=200, k=2000, kernel='gaussian'):
    # Theoretical density
    theoretical_fn = theoretical_density()
    x_grid = np.linspace(2, 12, 1000)

    # Generate sample
    sample = generate_mixture_sample(sample_size, seed=42)

    # Bandwidth estimate
    std = np.std(sample)
    bandwidth = 1.06 * std * sample_size ** (-1 / 5)

    # Method 1: KDE on original sample
    _, density_method1 = estimate_density(sample, kernel=kernel, bandwidth=bandwidth, x_grid=x_grid)
    mse_method1 = compute_mse(theoretical_fn, density_method1, x_grid)

    # Method 2: KDE on artificial sample
    artificial_sample = generate_artificial_sample(sample, k, bandwidth, seed=42)
    _, density_method2 = estimate_density(artificial_sample, kernel=kernel, bandwidth=bandwidth, x_grid=x_grid)
    mse_method2 = compute_mse(theoretical_fn, density_method2, x_grid)

    print(f"Method 1 MSE: {mse_method1:.5f}")
    print(f"Method 2 MSE: {mse_method2:.5f}")

    from plotting import plot_density_comparison_methods
    plot_density_comparison_methods(x_grid, theoretical_fn, density_method1, density_method2)
