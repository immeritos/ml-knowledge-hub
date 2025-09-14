import numpy as np

n = int(input().strip())
data = [float(input().strip()) for _ in range(n)]

def preprocess_data(data, missing=-1.0, round_digits=4):
    
    data = np.asarray(data)
    
    mean_value = np.mean(data[data != missing])
    
    data[data == -1] = mean_value
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    if iqr == 0:
        mask = np.ones_like(data, dtype=bool)
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (data >= lower) & (data <= upper)
        
    processed_data = np.round(data[mask], round_digits)
    
    return processed_data

# ========== sklearn version ==========
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

data = np.asarray(data).reshape(-1, 1)

data[data == -1] = np.nan

imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)

clf = IsolationForest(contamination=0.1, random_state=42)
outliers = clf.fit_predict(data_imputed)

processed_data = data_imputed[outliers == 1].flatten()