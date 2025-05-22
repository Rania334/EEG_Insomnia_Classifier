import pandas as pd
import numpy as np

def load_eeg_data(normal_path, diseased_path):
    X_normal = pd.read_csv(normal_path)
    X_diseased = pd.read_csv(diseased_path)

    y_normal = np.zeros(len(X_normal))
    y_diseased = np.ones(len(X_diseased))

    X = np.concatenate((X_normal, X_diseased), axis=0)
    y = np.concatenate((y_normal, y_diseased), axis=0)

    return X, y
