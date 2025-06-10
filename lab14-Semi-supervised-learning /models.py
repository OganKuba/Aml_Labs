# models.py
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.svm import SVC

def label_data(y_train, g, random_state=None):
    """
    Oznacza g przykładów klasy pozytywnej i g klasy negatywnej jako etykietowane,
    reszta to dane nieetykietowane (-1).
    """
    rng = np.random.RandomState(random_state)
    labeled_indices = []
    for cls in [0, 1]:
        cls_indices = np.where(y_train == cls)[0]
        selected = rng.choice(cls_indices, size=g, replace=False)
        labeled_indices.extend(selected)
    y_semi = np.full_like(y_train, fill_value=-1)
    y_semi[labeled_indices] = y_train[labeled_indices]
    return y_semi

def get_classifiers(base_estimator):
    """
    Zwraca słownik czterech modeli semi-supervised.
    """
    return {
        "Naive": base_estimator,
        "SelfTraining": SelfTrainingClassifier(base_estimator=base_estimator, criterion='k_best', k_best=2),
        "LabelPropagation": LabelPropagation(),
        "LabelSpreading": LabelSpreading(),
    }
