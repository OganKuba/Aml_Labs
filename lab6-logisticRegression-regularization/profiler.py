# Importowanie modułu pyplot z biblioteki matplotlib i nadanie mu aliasu 'plt'
import matplotlib.pyplot as plt


def plot_profile(lambdas, coefs, variable_names=None):
    """
    Funkcja rysuje profil współczynników (regularization path) dla różnych wartości lambda.

    Argumenty:
    lambdas (list or np.array): Lista wartości parametru regularyzacji lambda.
    coefs (np.array): Tablica NumPy, gdzie każdy wiersz odpowiada współczynnikom
                      modelu dla danej lambdy, a każda kolumna to jeden współczynnik.
    variable_names (list, optional): Lista nazw zmiennych (cech) do użycia w legendzie.
                                     Jeśli nie zostanie podana, zostaną użyte nazwy domyślne.
    """

    # Utworzenie nowego okna wykresu o określonym rozmiarze (szerokość 10 cali, wysokość 6 cali).
    plt.figure(figsize=(10, 6))

    # Pobranie liczby zmiennych (cech) z kształtu tablicy współczynników.
    # coefs.shape[1] to liczba kolumn, czyli liczba współczynników.
    num_vars = coefs.shape[1]

    # Pętla iterująca po każdej zmiennej (kolumnie w tablicy coefs).
    for i in range(num_vars):
        # Ustalenie etykiety dla linii na wykresie.
        # Jeśli lista `variable_names` została podana, użyj nazwy z tej listy.
        # W przeciwnym razie, utwórz etykietę domyślną, np. 'Var 1', 'Var 2', itd.
        var_label = variable_names[i] if variable_names is not None else f'Var {i + 1}'

        # Narysowanie wykresu dla i-tej zmiennej.
        # Na osi X są wartości lambda, na osi Y są wartości współczynników dla tej zmiennej.
        # `coefs[:, i]` wybiera wszystkie wiersze (dla każdej lambdy) dla i-tej kolumny (zmiennej).
        # `alpha=0.5` ustawia przezroczystość linii.
        plt.plot(lambdas, coefs[:, i], label=var_label, alpha=0.5)

    # Ustawienie skali logarytmicznej dla osi X, co jest standardem przy wizualizacji
    # ścieżek regularyzacji, ponieważ lambda często zmienia się o rzędy wielkości.
    plt.xscale('log')

    # Dodanie etykiet do osi X i Y oraz tytułu wykresu.
    plt.xlabel('Lambda (skala logarytmiczna)')
    plt.ylabel('Wartość współczynnika')
    plt.title('Ścieżka regularyzacji (Regularization Path)')

    # Dodanie legendy do wykresu.
    # `bbox_to_anchor` umieszcza legendę poza obszarem rysowania, aby nie zasłaniała danych.
    # `loc` określa punkt zaczepienia legendy.
    # `ncol` ustawia liczbę kolumn w legendzie.
    # `fontsize` kontroluje rozmiar czcionki w legendzie.
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)

    # Automatyczne dopasowanie elementów wykresu, aby uniknąć nakładania się etykiet.
    plt.tight_layout()

    # Wyświetlenie gotowego wykresu.
    plt.show()