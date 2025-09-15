import numpy as np

def shuffle_data_rd(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def shuffle_data_rs(X, y, seed=None):
    rng = np.random.RandomState(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], y[idx]

def shuffle_data_rng(X, y, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    return X[idx], y[idx]