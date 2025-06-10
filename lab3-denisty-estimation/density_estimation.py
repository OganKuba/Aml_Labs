import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# Funkcja generująca próbkę z mieszaniny dwóch rozkładów normalnych
def generate_mixture_sample(n, seed=None):
    np.random.seed(seed)  # ustawienie ziarna generatora losowego (dla powtarzalności wyników)

    # losujemy mieszankę: 90% szansa na 0, 10% szansa na 1
    mixture = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    # dla elementów ==0 generujemy liczby z N(5,1), a dla ==1 z N(10,1)
    sample = np.where(mixture == 0,
                      np.random.normal(5, 1, n),  # wartości z rozkładu N(5,1)
                      np.random.normal(10, 1, n))  # wartości z rozkładu N(10,1)

    return sample  # zwracamy wygenerowaną próbkę


# Funkcja zwracająca teoretyczną gęstość rozkładu mieszanego
def theoretical_density():
    # Zagnieżdżona funkcja, która oblicza gęstość w punkcie x
    def density(x):
        # 90% gęstości z rozkładu N(5,1) + 10% gęstości z N(10,1)
        return 0.9 * norm.pdf(x, loc=5, scale=1) + 0.1 * norm.pdf(x, loc=10, scale=1)

    return density  # zwraca funkcję density jako obiekt


def generate_artificial_sample(sample, k, bandwidth, seed=None):
    np.random.seed(seed)  # Ustawiamy ziarno generatora losowego, aby wyniki były powtarzalne (jeśli seed nie jest None)

    n = len(sample)  # Liczba elementów w oryginalnej próbce

    artificial_sample = []  # Lista do przechowywania sztucznie wygenerowanej próbki

    # Pętla, która wygeneruje k punktów
    for _ in range(k):
        i = np.random.choice(n)  # Losowo wybieramy indeks i z oryginalnej próbki (z przedziału od 0 do n-1)

        epsilon = np.random.normal(0, 1)  # Losujemy perturbację epsilon z rozkładu normalnego N(0,1)

        # Tworzymy nowy punkt: wybrany punkt + perturbacja * bandwidth
        artificial_sample.append(sample[i] + epsilon * bandwidth)

    # Konwertujemy listę na tablicę numpy i zwracamy sztuczną próbkę
    return np.array(artificial_sample)


# Funkcja szacująca gęstość rozkładu na podstawie próbki
def estimate_density(sample, kernel="gaussian", bandwidth=None, x_grid=None):
    # Tworzymy siatkę punktów od 2 do 12 (1000 punktów), w kolumnowej macierzy (1000x1)
    if x_grid is None:
        x_grid = np.linspace(2, 12, 1000).reshape(-1, 1)
    else:
        x_grid = x_grid.reshape(-1, 1)

    # Jeśli użytkownik nie podał bandwidth, to obliczamy go wg reguły Silvermana
    if bandwidth is None:
        std = np.std(sample)  # odchylenie standardowe próbki
        n = len(sample)  # liczba obserwacji
        bandwidth = 1.06 * std * n ** (-1 / 5)  # reguła Silvermana na bandwidth

    # Tworzymy obiekt KernelDensity z podanym jądrem i szerokością pasma
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    # Dopasowujemy estymator do danych (konwertujemy 1D wektor sample na macierz kolumnową)
    kde.fit(sample[:, np.newaxis])

    # Obliczamy logarytm gęstości w punktach x_grid
    log_density = kde.score_samples(x_grid)

    # UWAGA: w Twoim kodzie density_estimate = kde.score_samples(x_grid) robi to samo co log_density
    # Powinno być density_estimate = np.exp(log_density) aby uzyskać rzeczywistą gęstość
    density_estimate = np.exp(log_density)  # przekształcamy logarytmy gęstości na gęstość

    # Zwracamy siatkę punktów (jako wektor 1D) i oszacowane wartości gęstości
    return x_grid.flatten(), density_estimate
