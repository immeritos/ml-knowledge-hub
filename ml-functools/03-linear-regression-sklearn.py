from typing import Tuple, List
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

rng = np.random.RandomState(42)
X = rng.uniform(0, 10, size=(100, 2))
true_w = np.array([1.5, -3.0])
y = 0.8 + X @ true_w + rng.normal(0, 0.5, size=100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ======= LinearRegression ========

linreg = LinearRegression(
    fit_intercept=True,
    copy_X = True,
    n_jobs=None,
    positive=False
)
linreg.fit(X_train, y_train)

print("=== LinearRegression (SVD) ===")
print("intercept:", float(linreg.intercept_))
print("coef_:", linreg.coef_.tolist())
print("R^2 on train:", linreg.score(X_train, y_train))
print("R^2 on test:", linreg.score(X_test, y_test))

cv_scores = cross_val_score(linreg, X, y, cv=5, scoring="r2")
print("CV R^2 mean +- std:", float(cv_scores.mean()), float(cv_scores.std()))

# ======= Pipeline + SGDRegressor ========
sgd_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        l1_ratio=0.15,
        max_iter=2000,
        tol=1e-6,
        learning_rate="optimal",
        eta0=0.01,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        fit_intercept=True
    ))
])

sgd_pipeline.fit(X_train, y_train)

sgd = sgd_pipeline.named_steps["sgd"]
print("\n=== SGDRegressor (with StandardScaler) ===")
print("intercept_:", float(sgd.intercept_[0]))
print("coef_:", sgd.coef_.tolist())
print("R^2 on train:", sgd_pipeline.score(X_train, y_train))
print("R^2 on test :", sgd_pipeline.score(X_test, y_test))