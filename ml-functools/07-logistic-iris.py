from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import random


X, y = load_iris(return_X_y=True)

n = int(input())

rng = np.random.default_rng(42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

X_train, X_test = X[:-n], X[-n:]
y_train, y_test = y[:-n], y[-n:]

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
target_names = load_iris().target_names # type:ignore

for i in range(len(y_pred)):
    print(f'{target_names[y_pred[i]]} {np.max(y_prob[i]):.2f}')
