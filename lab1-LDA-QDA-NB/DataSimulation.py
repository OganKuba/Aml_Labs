from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from Generator import accuracy_score
import numpy as np
import pandas as pd
from QDA import QDA
from LDA import LDA
from NB import NB
from sklearn.preprocessing import LabelEncoder


def load_openml_dataset(data_id):
    data = fetch_openml(data_id=data_id, as_frame=False)
    X = data.data
    y = data.target

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded


def compare_methods(X, y, n_splits=25, test_size=0.3, random_seed=123):
    results = []
    np.random.seed(random_seed)

    for _ in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=np.random.randint(1e9)
        )

        lda = LDA()
        lda.fit(X_train, y_train)
        acc_lda = accuracy_score(y_test, lda.predict(X_test))
        results.append(["LDA", acc_lda])

        qda = QDA()
        qda.fit(X_train, y_train)
        acc_qda = accuracy_score(y_test, qda.predict(X_test))
        results.append(["QDA", acc_qda])

        nb = NB()
        nb.fit(X_train, y_train)
        acc_nb = accuracy_score(y_test, nb.predict(X_test))
        results.append(["NB", acc_nb])

    df = pd.DataFrame(results, columns=["Method", "Accuracy"])
    return df


def data_simulation():
    datasets = {
        "OpenML 995 - mfeat-zernike": lambda: load_openml_dataset(995),
        "OpenML 979 - waveform-5000": lambda: load_openml_dataset(979),
        "OpenML 847 - wind": lambda: load_openml_dataset(847),
    }

    all_results = {}

    for dataset_name, loader in datasets.items():
        X, y = loader()
        df_results = compare_methods(X, y, n_splits=25, test_size=0.3)
        all_results[dataset_name] = df_results

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey=True)

    for i, (dataset_name, df_results) in enumerate(all_results.items()):
        ax = axs[i]
        df_results.boxplot(column="Accuracy", by="Method", ax=ax, grid=False)
        ax.set_title(dataset_name)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("BayesianReal.pdf")
    plt.close()
