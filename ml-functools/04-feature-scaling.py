import numpy as np

def manual_standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data.tolist()

def manual_minmax(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.tolist()

# ======= sklearn ========
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.array([[1.0, 1000.0],
              [2.0, 1500.0],
              [3.0, 2000.0]])

scaler_std = StandardScaler()
standardized_data = scaler_std.fit_transform(data)

scaler_mm = MinMaxScaler()
normalized_data = scaler_mm.fit_transform(data)