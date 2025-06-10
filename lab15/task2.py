import numpy as np
import pandas as pd

def generate_X1(n):
    """
    Generuje cechę X1 z mieszanki 3 rozkładów normalnych:
    0.25*N(-1, 0.2^2) + 0.25*N(1, 0.2^2) + 0.5*N(5, 0.2^2)
    """

    components = np.random.choice([0, 1, 2], size=n, p=[0.25, 0.25, 0.5])

    means = np.array([-1, 1, 5])
    sds = np.array([0.2, 0.2, 0.2])

    X1 = np.random.normal(loc=means[components], scale=sds[components])

    return X1

def generate_X_rest(n):
    """
    Generuje cechy X2,...,X10 ~ N(0,1)
    """

    X_rest = np.random.normal(loc=0, scale=1, size=(n, 9))
    return X_rest

def generate_Y(X1):
    """
        Generuje zmienną Y:
        P(Y=1|X) = 0.1 gdy X1<3
                 = 0.9 gdy X1>=3
    """
    probs = np.where(X1 < 3, 0.1, 0.9)
    Y = np.random.binomial(n=1, p=probs)
    return Y

def generate_dataset(n):
    """
    Generuje zbiór danych o rozmiarze n.
    """
    X1 = generate_X1(n)
    X_rest = generate_X_rest(n)

    X = np.column_stack([X1, X_rest])
    Y = generate_Y(X1)

    return X, Y

def main():
    # Rozmiar zbiorów
    n_train = 1000
    n_test = 1000

    # Generowanie danych treningowych
    X_train, Y_train = generate_dataset(n_train)
    # Konwersja na DataFrame (opcjonalnie)
    columns = [f'X{i+1}' for i in range(10)]
    df_train = pd.DataFrame(X_train, columns=columns)
    df_train['Y'] = Y_train

    # Generowanie danych testowych
    X_test, Y_test = generate_dataset(n_test)
    df_test = pd.DataFrame(X_test, columns=columns)
    df_test['Y'] = Y_test

    # Wyświetlenie kilku wierszy dla podglądu
    print("Przykładowe dane treningowe:")
    print(df_train.head())

    print("\nPrzykładowe dane testowe:")
    print(df_test.head())

    # (opcjonalnie) zapis do CSV
    df_train.to_csv("train_data.csv", index=False)
    df_test.to_csv("test_data.csv", index=False)
    print("\nDane zapisane do plików train_data.csv i test_data.csv")

if __name__ == "__main__":
    main()
