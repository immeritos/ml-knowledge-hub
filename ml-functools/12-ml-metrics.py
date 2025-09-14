import numpy as np

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_pred == y_pred, axis=0) / len(y_true)
    return round(accuracy, 3)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return round(precision, 3)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(recall, 3)

def f_score(precision, recall, beta):
    op = precision * recall
    div = ((beta**2) * precision) + recall
    
    if div == 0 or op == 0:
        return 0.0
    score = (1 + (beta ** 2)) * op / div
    return round(score, 3)

