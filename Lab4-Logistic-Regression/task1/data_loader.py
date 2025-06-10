import pandas as pd  # Importujemy bibliotekę pandas do pracy z danymi tabelarycznymi

def load_earthquake_data(file_path):
    # Wczytujemy dane z pliku CSV
    # - file_path: ścieżka do pliku tekstowego
    # - sep=r"\s+": separator - jeden lub więcej białych znaków (np. spacja, tabulator)
    # - header=0: pierwszy wiersz zawiera nagłówki kolumn
    # - dtype=str: wczytujemy wszystkie dane jako stringi, żeby na początku uniknąć problemów z typami danych
    df = pd.read_csv(file_path, sep=r"\s+", header=0, dtype=str)

    # Mapujemy kolumnę 'popn' (klasę) na wartości binarne:
    # - 'equake' -> 0 (trzęsienie ziemi)
    # - 'explosn' -> 1 (wybuch)
    df['popn'] = df['popn'].map({'equake': 0, 'explosn': 1})

    # Usuwamy wiersze, gdzie brakuje wartości w kolumnie 'popn'
    df = df.dropna(subset=['popn'])

    # Konwertujemy kolumnę 'popn' na typ int (bo wcześniej była jako string)
    df['popn'] = df['popn'].astype(int)

    # Konwertujemy kolumny 'body' i 'surface' na liczby zmiennoprzecinkowe (float)
    df[['body', 'surface']] = df[['body', 'surface']].astype(float)

    # Zwracamy przetworzony DataFrame
    return df
