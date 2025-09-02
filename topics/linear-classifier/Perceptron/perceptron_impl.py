import numpy as np

class Perceptron:
    """
    Standard perceptron with optional weight averaging (averaged=True).
    Converges on linearly separable data; otherwise performs online updates up to max_iter.
    """

    def __init__(self, lr=1.0, max_iter=20, shuffle=True, averaged=False, random_state=None):
        self.lr = lr
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.averaged = averaged
        self.random_state = np.random.RandomState(random_state)
        self.w_ = None
        self.b_ = 0.0

    @staticmethod
    def _ensure_pm1_labels(y):
        y = np.asarray(y)
        uniq = np.unique(y)
        if set(uniq) == {-1, 1}:
            return y.astype(float)
        if len(uniq) != 2:
            raise ValueError("Binary labels required.")
        # Safe mapping: missing keys raise KeyError instead of silently returning None
        y_map = {uniq[0]: -1.0, uniq[1]: 1.0}
        return np.vectorize(y_map.__getitem__)(y).astype(float)

    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._ensure_pm1_labels(y)
        n_samples, n_features = X.shape

        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0

        if self.averaged:
            w_sum = np.zeros_like(self.w_)
            b_sum = 0.0
            count = 0

        for _ in range(self.max_iter):
            idx = np.arange(n_samples)
            if self.shuffle:
                self.random_state.shuffle(idx)

            errors = 0
            for i in idx:
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w_, xi) + self.b_) <= 0:
                    self.w_ += self.lr * yi * xi
                    self.b_ += self.lr * yi
                    errors += 1
                if self.averaged:
                    w_sum += self.w_
                    b_sum += self.b_
                    count += 1

            if errors == 0:
                break

        if self.averaged and count > 0:
            self.w_ = w_sum / count
            self.b_ = b_sum / count
        return self
    
class PocketPerceptron:
    """
    Perceptron with Pocket: keeps the parameters (w*, b*) that yield the fewest
    training errors seen so far. Useful on nonseparable/noisy data.
    """

    def __init__(self, lr=1.0, max_iter=50, shuffle=True, per_update_check=False, random_state=None):
        self.lr = lr
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.per_update_check = per_update_check
        self.random_state = np.random.RandomState(random_state)
        self.w_ = None
        self.b_ = 0.0
        self.w_best_ = None
        self.b_best_ = 0.0
        self.best_errors_ = np.inf

    @staticmethod
    def _ensure_pm1_labels(y):
        y = np.asarray(y)
        uniq = np.unique(y)
        if set(uniq) == {-1, 1}:
            return y.astype(float)
        if len(uniq) != 2:
            raise ValueError("Binary labels required.")
        y_map = {uniq[0]: -1.0, uniq[1]: 1.0}
        return np.vectorize(y_map.__getitem__)(y).astype(float)

    def decision_function(self, X, w=None, b=None):
        if w is None: w = self.w_
        if b is None: b = self.b_
        return X @ w + b

    def _count_errors(self, X, y, w=None, b=None):
        y_pred = np.where(self.decision_function(X, w, b) >= 0, 1.0, -1.0)
        return int(np.sum(y_pred != y))

    def predict(self, X, use_pocket=True):
        if use_pocket and self.w_best_ is not None:
            s = self.decision_function(X, self.w_best_, self.b_best_)
        else:
            s = self.decision_function(X, self.w_, self.b_)
        return np.where(s >= 0, 1.0, -1.0)

    def score(self, X, y):
        y = self._ensure_pm1_labels(y)
        return float(np.mean(self.predict(X, use_pocket=True) == y))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._ensure_pm1_labels(y)
        n_samples, n_features = X.shape

        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0

        self.w_best_ = self.w_.copy()
        self.b_best_ = float(self.b_)
        self.best_errors_ = self._count_errors(X, y, self.w_best_, self.b_best_)

        for _ in range(self.max_iter):
            idx = np.arange(n_samples)
            if self.shuffle:
                self.random_state.shuffle(idx)
            errors_epoch = 0

            for i in idx:
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w_, xi) + self.b_) <= 0:
                    self.w_ += self.lr * yi * xi
                    self.b_ += self.lr * yi
                    errors_epoch += 1

                    if self.per_update_check:
                        cur = self._count_errors(X, y, self.w_, self.b_)
                        if cur < self.best_errors_:
                            self.best_errors_ = cur
                            self.w_best_ = self.w_.copy()
                            self.b_best_ = float(self.b_)
                            if self.best_errors_ == 0:
                                self.w_, self.b_ = self.w_best_, self.b_best_
                                return self

            if not self.per_update_check:
                cur = self._count_errors(X, y, self.w_, self.b_)
                if cur < self.best_errors_:
                    self.best_errors_ = cur
                    self.w_best_ = self.w_.copy()
                    self.b_best_ = float(self.b_)

            if errors_epoch == 0:
                self.w_, self.b_ = self.w_best_, self.b_best_
                return self

        self.w_, self.b_ = self.w_best_, self.b_best_
        return self