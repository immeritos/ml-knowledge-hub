import numpy as np

class LogisticRegression:
    
    def __init__(self, lr=0.1, max_iter=1000, l2=0.0, tol=1e-6, standardize=True, random_state=42):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.tol = tol
        self.standardize = standardize
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None
        
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def _softplus(x):
        x_abs = np.abs(x)
        return np.maximum(x, 0) + np.log1p(np.exp(-x_abs))
    
    def _standardize_fit(self, X):
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_[self.sigma_ == 0] = 1.0
    
    def _standardize_transfrom(self, X):
        return (X - self.mu_) / self.sigma_
    
    def _loss(self, X, y):
        z = X @ self.w + self.b
        loss_ce = self._softplus(z) - y * z
        loss = loss_ce.mean()
        if self.l2 > 0:
            loss += 0.5 * self.l2 * np.dot(self.w, self.w)
        return float(loss)
    
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        
        if self.standardize:
            self._standardize_fit(X)
            X = self._standardize_transfrom(X)
            
        m, d = X.shape
        self.w = rng.normal(0, 0.01, size=d)
        self.b = 0.0
        
        prev_loss = np.inf
        for iter in range(1, self.max_iter + 1):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            e = p - y
            
            grad_w = (X.T @ e) / m
            grad_b = e.mean()
            
            if self.l2 > 0:
                grad_w += self.l2 * self.w
                
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            curr_loss = self._loss(X, y)
            if abs(prev_loss - curr_loss) < self.tol:
                break
            prev_loss = curr_loss
        
        return self
    
    def predict_proba(self, X):
        if self.w is None:
            raise RuntimeError("Modal not fitted.")
        if self.standardize:
            X = self._standardize_transfrom(X)
        z = X @ self.w + self.b
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
    
