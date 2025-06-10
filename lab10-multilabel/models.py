from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_binary_relevance(X_train, Y_train):
    """
    Trains one independent classifier per label.
    If a label has only one class, trains a dummy classifier.
    """
    models = []
    for k in range(Y_train.shape[1]):
        y_k = Y_train[:, k]
        unique_classes = np.unique(y_k)
        if len(unique_classes) == 1:
            # Use a DummyClassifier that always predicts that single class
            clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
        else:
            clf = LogisticRegression(max_iter=5000)
        clf.fit(X_train, y_k)
        models.append(clf)
    return models

def predict_binary_relevance(models, X_test):
    """
    Predicts labels for test data using trained BR models.
    """
    for idx, model in enumerate(models):
        print(f"Model {idx}: type = {type(model)}")
    Y_pred = np.array([model.predict(X_test) for model in models]).T
    return Y_pred

def train_classifier_chain(X_train, Y_train, order=None):
    """
    Trains a single classifier chain model.
    Handles single-class columns with DummyClassifier.
    """
    base_clf = LogisticRegression(max_iter=5000)
    n_labels = Y_train.shape[1]
    valid_labels = []
    dummy_labels = {}
    chain_models = []

    for k in range(n_labels):
        y_k = Y_train[:, k]
        unique = np.unique(y_k)
        if len(unique) == 1:
            # Single-class column; store dummy
            dummy_labels[k] = unique[0]
        else:
            valid_labels.append(k)

    if len(valid_labels) == 0:
        # all columns are single-class! Return dummy info
        return {"dummies": dummy_labels, "chain": None}

    # Train chain on valid columns
    chain = ClassifierChain(base_estimator=base_clf, order=order, random_state=42)
    chain.fit(X_train, Y_train[:, valid_labels])

    return {"chain": chain, "dummies": dummy_labels, "valid_labels": valid_labels}

def predict_classifier_chain(model_dict, X_test):
    """
    Predicts labels using a trained classifier chain model, plus dummies.
    """
    n_samples = X_test.shape[0]
    n_labels = max(model_dict["dummies"].keys(), default=-1) + 1
    if model_dict["chain"] is not None:
        Y_partial = model_dict["chain"].predict(X_test)
        Y_pred = np.zeros((n_samples, n_labels), dtype=int)
        # Insert predicted labels into correct columns
        for idx, label_idx in enumerate(model_dict["valid_labels"]):
            Y_pred[:, label_idx] = Y_partial[:, idx]
    else:
        Y_pred = np.zeros((n_samples, n_labels), dtype=int)

    # Fill dummy columns
    for label_idx, constant_value in model_dict["dummies"].items():
        Y_pred[:, label_idx] = constant_value

    return Y_pred

def train_ecc(X_train, Y_train, n_chains=5):
    """
    Trains an ensemble of classifier chains with different random orders.
    Handles single-class columns with DummyClassifier logic.
    """
    ensemble = []
    base_clf = LogisticRegression(max_iter=5000)

    n_labels = Y_train.shape[1]

    for i in range(n_chains):
        dummy_labels = {}
        valid_labels = []

        for k in range(n_labels):
            y_k = Y_train[:, k]
            unique = np.unique(y_k)
            if len(unique) == 1:
                dummy_labels[k] = unique[0]
            else:
                valid_labels.append(k)

        if len(valid_labels) == 0:
            # all columns are single-class! Skip this chain
            ensemble.append({"chain": None, "dummies": dummy_labels, "valid_labels": []})
            continue

        chain = ClassifierChain(base_estimator=base_clf, order='random', random_state=i)
        chain.fit(X_train, Y_train[:, valid_labels])
        ensemble.append({"chain": chain, "dummies": dummy_labels, "valid_labels": valid_labels})

    return ensemble


def predict_ecc(ensemble, X_test):
    """
    Predicts labels using an ensemble of classifier chains with dummy support.
    """
    n_samples = X_test.shape[0]
    n_labels = max(
        max(chain_data["dummies"].keys(), default=-1) for chain_data in ensemble
    ) + 1
    Y_preds = []

    for chain_data in ensemble:
        Y_pred = np.zeros((n_samples, n_labels), dtype=int)
        if chain_data["chain"] is not None:
            Y_partial = chain_data["chain"].predict(X_test)
            for idx, label_idx in enumerate(chain_data["valid_labels"]):
                Y_pred[:, label_idx] = Y_partial[:, idx]
        for label_idx, constant_value in chain_data["dummies"].items():
            Y_pred[:, label_idx] = constant_value
        Y_preds.append(Y_pred)

    Y_preds = np.array(Y_preds)
    Y_mean = Y_preds.mean(axis=0)
    return (Y_mean >= 0.5).astype(int)
