import pandas as pd

def load_prostate_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(columns=['label'])
    y = data['label']
    return X, y