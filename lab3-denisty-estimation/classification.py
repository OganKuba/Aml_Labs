import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB, BernoulliNB


def load_data():
    from sklearn.datasets import load_breast_cancer  # importujemy zbiór danych Breast Cancer Wisconsin z scikit-learn

    data = load_breast_cancer()  # wczytujemy dane: to obiekt typu Bunch z cechami i etykietami

    return data.data, data.target  # zwracamy cechy (X) i etykiety (y) w formie numpy arrays


def train_test_split_data(X, y, test_size=0.3, seed=42):
    # Funkcja dzieli dane na zbiór treningowy i testowy

    # train_test_split pochodzi z biblioteki sklearn.model_selection
    # - X: macierz cech (np. dane wejściowe)
    # - y: wektor etykiet (klasy)
    # - test_size=0.3: 30% danych idzie do zbioru testowego, 70% do treningowego
    # - random_state=seed: ustawienie ziarna losowości (powtarzalne wyniki)
    return train_test_split(X, y, test_size=test_size, random_state=seed)

class KDENaiveBayes:
    def __init__(self, bandwidth=None, kernel='gaussian'):
        # Konstruktor klasy: ustawia domyślny bandwidth i kernel
        self.bandwidth = bandwidth  # szerokość pasma jądra
        self.kernel = kernel  # rodzaj jądra (domyślnie 'gaussian')

    def fit(self, X, y):
        # Dopasowanie modelu do danych treningowych X i etykiet y
        self.classes_ = np.unique(y)  # lista unikalnych klas
        self.models_ = {}  # słownik na modele KDE dla każdej klasy
        self.priors_ = {}  # słownik na prawdopodobieństwa a priori

        # Iterujemy po każdej klasie w zbiorze treningowym
        for c in self.classes_:
            Xc = X[y == c]  # wybieramy tylko próbki należące do klasy c
            self.priors_[c] = Xc.shape[0] / X.shape[0]  # estymujemy prior (częstość klasy)

            self.models_[c] = []  # lista na modele KDE dla każdej cechy

            # Iterujemy po wszystkich cechach (kolumnach) w X
            for j in range(X.shape[1]):
                # Obliczamy bandwidth dla danej cechy w danej klasie
                bw = self.bandwidth or (1.06 * np.std(Xc[:, j]) * len(Xc)**(-1/5))
                # Tworzymy i dopasowujemy KDE dla tej cechy i klasy
                kde = KernelDensity(kernel=self.kernel, bandwidth=bw).fit(Xc[:, j:j + 1])
                self.models_[c].append(kde)  # dodajemy model KDE do listy

    def predict(self, X):
        # Predykcja klasy dla danych X
        log_probs = np.zeros((len(X), len(self.classes_)))  # macierz na logarytmy gęstości

        # Iterujemy po klasach i obliczamy logarytmy prawdopodobieństw a posteriori
        for idx, c in enumerate(self.classes_):
            lp = np.log(self.priors_[c])  # logarytm priora klasy
            for j in range(X.shape[1]):
                # Obliczamy logarytm gęstości dla danej cechy i klasy
                log_d = self.models_[c][j].score_samples(X[:, j:j + 1])
                lp += log_d  # dodajemy (logarytmy) gęstości dla wszystkich cech
            log_probs[:, idx] = lp  # zapisujemy wynik dla danej klasy

        # Zwracamy etykietę klasy z największym prawdopodobieństwem a posteriori
        return self.classes_[np.argmax(log_probs, axis=1)]

def discretize_features(X, n_bins=10):
    # Funkcja dyskretyzująca zmienne ilościowe (ciągłe) na zmienne kategoryczne (bins)
    # Argumenty:
    # - X: macierz danych (np. cechy)
    # - n_bins: liczba przedziałów (binów) do podziału każdej cechy (domyślnie 10)

    Xb = np.zeros_like(X)  # Tworzymy nową macierz (o tych samych wymiarach co X) wypełnioną zerami

    # Iterujemy po wszystkich kolumnach (cechach) w X
    for j in range(X.shape[1]):
        # Wyznaczamy granice przedziałów (bins) równomiernie rozmieszczonych między min i max wartości w kolumnie j
        bins = np.linspace(np.min(X[:, j]), np.max(X[:, j]), n_bins + 1)

        # Dla każdej wartości w kolumnie j przypisujemy numer przedziału (bin)
        # np.digitize przypisuje binom numery od 1 do n_bins+1, więc odejmujemy 1 żeby mieć od 0 do n_bins
        Xb[:, j] = np.digitize(X[:, j], bins) - 1

    # Zwracamy zdyskretyzowaną macierz cech (każda kolumna zawiera teraz numery binów)
    return Xb


def evaluate_all():
    # Wczytujemy dane (cechy i etykiety) z funkcji load_data()
    X, y = load_data()

    # Dzielimy dane na treningowe i testowe (np. 70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    results = {}  # Słownik na wyniki dokładności dla wszystkich metod

    # 1️⃣ KDE Naive Bayes
    kde_nb = KDENaiveBayes()  # Tworzymy obiekt KDE Naive Bayes
    kde_nb.fit(X_train, y_train)  # Dopasowujemy model do danych treningowych
    # Obliczamy accuracy na zbiorze testowym
    results['KDE-NB'] = accuracy_score(y_test, kde_nb.predict(X_test))

    # 2️⃣ GaussianNB
    gnb = GaussianNB()  # Tworzymy obiekt Gaussian Naive Bayes (wbudowany w sklearn)
    gnb.fit(X_train, y_train)  # Dopasowujemy model do danych treningowych
    # Obliczamy accuracy na zbiorze testowym
    results['GaussianNB'] = accuracy_score(y_test, gnb.predict(X_test))

    # 3️⃣ Dyskretyzacja + BernoulliNB
    # Dyskretyzujemy dane treningowe (np. zamieniamy cechy ciągłe na binarne lub kategoryczne)
    Xb_train = discretize_features(X_train)
    Xb_test = discretize_features(X_test)
    # Tworzymy obiekt BernoulliNB (dla danych binarnych)
    bnb = BernoulliNB()
    bnb.fit(Xb_train, y_train)  # Dopasowujemy model do danych treningowych
    # Obliczamy accuracy na zbiorze testowym
    results['Disc-NB'] = accuracy_score(y_test, bnb.predict(Xb_test))

    # 4️⃣ LDA
    lda = LinearDiscriminantAnalysis()  # Tworzymy obiekt LDA (Linear Discriminant Analysis)
    lda.fit(X_train, y_train)  # Dopasowujemy model do danych treningowych
    # Obliczamy accuracy na zbiorze testowym
    results['LDA'] = accuracy_score(y_test, lda.predict(X_test))

    # Zwracamy słownik z wynikami wszystkich metod
    return results





