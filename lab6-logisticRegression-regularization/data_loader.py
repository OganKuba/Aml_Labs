import pandas as pd


def load_prostate_data(x_filepath, y_filepath):
    """
    Wczytuje dane z osobnych plików tekstowych, dostosowując się do specyficznej
    struktury plików, gdzie separatorem jest spacja, a pliki zawierają nagłówki.

    Argumenty:
    x_filepath (str): Ścieżka do pliku z cechami (X).
    y_filepath (str): Ścieżka do pliku z etykietami (y).

    Zwraca:
    tuple: Krotka zawierająca:
           - X (pd.DataFrame): Ramka danych z cechami.
           - y (pd.Series): Seria danych z etykietami.
    """

    # --- Wczytanie cech (X) ---
    X = pd.read_csv(x_filepath, sep='\s+')

    # --- Wczytanie etykiet (y) ---

    y = pd.read_csv(y_filepath, sep='\s+')['x']

    return X, y
