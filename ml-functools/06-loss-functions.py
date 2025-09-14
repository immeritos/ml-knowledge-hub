import numpy as np

def calculate_loss(y_true, y_pred, delta):
    """
    calculate MSE / MAE / Huber / Cosine loss
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    y_mean = np.mean(y_true)
    
    r = y_true - y_pred
    
    ssr = np.sum(r ** 2)
    sst = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ssr / sst)
    
    mse = np.mean(r ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(r))
    
    huber_loss = np.where(
        np.abs(r) <= delta, 
        0.5 * (r ** 2), 
        delta * (np.abs(r) - 0.5 * delta)
    )
    huber_mean = float(np.mean((huber_loss)))
    
    y_true_norm = np.linalg.norm(y_true)
    y_pred_norm = np.linalg.norm(y_pred)
    if y_true_norm == 0.0 or y_pred_norm == 0.0:
        cosine_loss = 0.0
    else:
        cosine_loss = 1 - float(np.dot(y_true, y_pred) / (y_true_norm * y_pred_norm))
    return r2, mse, rmse, mae, huber_mean, cosine_loss

n = int(input())
y_true = []
y_pred = []

for _ in range(n):
    real, pred = map(float, input().split())
    y_true.append(real)
    y_pred.append(pred)
    
delta = float(input())

results = calculate_loss(y_true, y_pred, delta)
for value in results:
    print(f"{value:.6f}")
    
# ========== sklearn version ==========
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

y_true = np.asarray(y_true, dtype=float)
y_pred = np.asarray(y_pred, dtype=float)
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

cosine_loss = 1.0 - float(cosine_similarity(y_true.reshape(1, -1), y_pred)[0, 0])