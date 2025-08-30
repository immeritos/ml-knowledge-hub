# Perceptron

> Scope: intuition → math → code (averaged & pocket) → Python idioms from the code → variants (SVM, MLP).

---

## 1. The Perceptron: What/Why/How

### 1.1 Definition & Goal
- **Perceptron** is an early **linear binary classifier** (Rosenblatt, 1958).  
- It learns a separating hyperplane
  $$
  f(x)=\operatorname{sign}(w^\top x + b),\quad y\in\{-1,+1\}.
  $$
- **Why**: When classes are (approximately) linearly separable, it’s a fast, simple baseline with online/streaming-friendly updates.

### 1.2 Working Principle (Error-Driven / Online)
- **Decision**: $\hat y=\operatorname{sign}(w^\top x+b)$.
- **Update only when misclassified** (error-driven):
 $$
  \text{if } y_i\,(w^\top x_i+b)\le 0:\quad
  w\leftarrow w+\eta\,y_i x_i,\;\; b\leftarrow b+\eta\,y_i,
 $$
  where $\eta>0$ is the learning rate.
- **View as SGD**: It’s equivalent to stochastic optimization of the **perceptron loss** $\max(0,-y\,f(x))$ (a 0–1 loss surrogate).

### 1.3 Convergence & Geometry (intuition)
- If data are **linearly separable**, the perceptron makes a **finite** number of mistakes and stops (mistake bound depends on margin $\gamma$ and radius $R$).  
- Each update nudges the hyperplane **toward the correct side**, akin to making progress in the direction of a feasible separator.

### 1.4 Common Gotchas
- **Won’t converge** on nonseparable data → it may oscillate; use **Averaged** or **Pocket** variants or add regularization/early stopping.  
- **No probabilities**: outputs are hard labels; use **logistic regression** (with sigmoid/softmax) if you need calibrated probabilities.  
- **Order matters**: sample order affects the solution; always **shuffle** per epoch.

### 1.5 Quick Contrast
- **Perceptron vs Logistic Regression**: hard decisions vs probability outputs (maximizes log-likelihood).  
- **Perceptron vs SVM**: “just separate” vs **maximize margin** (hinge loss + regularization, typically more robust).

---

## 2. Implementations (Averaged & Pocket Perceptron)

> Requirements: `numpy`. Labels may be `{0,1}` or `{-1,+1}`; we’ll map to `{-1,+1}` internally.

### 2.1 Averaged-Capable Perceptron

```python
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

```

#### Mini Demo
```python
X = np.array([[2,3],[4,5],[1,0],[2,1],[2.2,1.1],[1.8,3.2],[3.5,4.1]], dtype=float)
y = np.array([ 1,   1,   -1,   -1,    -1,        1,        1])
pp = PocketPerceptron(lr=1.0, max_iter=50, random_state=0).fit(X, y)
print("Pocket best errors:", pp.best_errors_)
print("w*, b* =", pp.w_best_, pp.b_best_)
print("train acc:", pp.score(X, y))
```

---

### 2.2 Pocket Perceptron (best-in-training snapshot)

```python
import numpy as np

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
```

#### Mini Demo
```python
X = np.array([[2,3],[4,5],[1,0],[2,1],[2.2,1.1],[1.8,3.2],[3.5,4.1]], dtype=float)
y = np.array([ 1,   1,   -1,   -1,    -1,        1,        1])
pp = PocketPerceptron(lr=1.0, max_iter=50, random_state=0).fit(X, y)
print("Pocket best errors:", pp.best_errors_)
print("w*, b* =", pp.w_best_, pp.b_best_)
print("train acc:", pp.score(X, y))
```

---

## 3. Python Building Blocks Illustrated by the Code

### 3.1 RNG: `random_state` (param) vs `self.random_state` (attribute)
- Pass a seed or RNG via the constructor (e.g., `random_state=0`) and **store a dedicated RNG** inside the model for reproducibility and isolation from the global RNG:

```
python
self.random_state = np.random.RandomState(seed)
# or modern API:
# self.rng = np.random.default_rng(seed)
```

- Typical shuffling patterns:

```python
idx = self.random_state.permutation(n_samples)  # returns a shuffled index array
# or
idx = np.arange(n_samples); self.random_state.shuffle(idx)  # in-place
```

- Prefer keeping a model-local RNG attribute (e.g., `self.random_state` or `self.rng`) so multiple model instances don’t interfere.

### 3.2 `np.unique` for label checking/mapping
- Purpose: **deduplicate and sort** labels, verify binary classification, and map to `{-1,+1}` internally.
- Safe mapping (raises on unknown labels) using `__getitem__`:
```python
uniq = np.unique(y)
y_map = {uniq[0]: -1.0, uniq[1]: 1.0}
y = np.vectorize(y_map.__getitem__)(y).astype(float)

```
- Fast two-class shortcut: for labels `a` and `b`, map with a boolean mask (e.g., `y == a → -1.0`, else `+1.0`).

### 3.3 `np.zeros` vs `np.zeros_like`
- `np.zeros(shape, dtype=...)`: create a zero array with an **explicit shape** (default dtype `float64`).
- `np.zeros_like(a, dtype=...)`: create a zero array that **matches `a`’s shape/dtype/layout**—ideal for accumulators aligned with existing arrays:
```
python
w_sum = np.zeros_like(self.w_)
```

### 3.4 Underscore naming conventions
- **Trailing underscore** (e.g., `w_`, `b_`, `coef_`, `intercept_`): denotes **learned attributes** available **after** `fit`, following scikit-learn style.
- **Leading underscore** (e.g., `_helper`): marks **non-public** helpers/internal APIs (conventional privacy; still accessible but not part of the stable interface).
- Avoid keyword collisions with a trailing underscore (e.g., `class_`, `id_`).  
- Dunder methods (e.g., `__init__`, `__call__`) are **special hooks**; double-leading underscores (`__name`) trigger **name mangling** (rarely needed for ML models).

### 3.5 `@staticmethod` in class code
- Lives in the class **namespace** but **does not receive `self`/`cls`**.  
- Use when logic is conceptually tied to the class but **doesn’t need instance or class state**.  
- If you need class-level context or polymorphism, prefer `@classmethod`; if you need instance data, use an instance method.

### 3.6 Why multiple epochs & shuffling?
- Multiple epochs **reduce variance** of noisy per-sample updates (SGD) and let examples influence the model in **different parameter states**.  
- Always **shuffle per epoch** to break harmful order correlations:
```python
for epoch in range(max_iter):
    idx = rng.permutation(n_samples)
    for i in idx: ...
```
- **Early stopping**: on linearly separable data, stop when an epoch sees zero mistakes; on noisy/nonseparable data, monitor validation metrics to avoid overfitting.

---

## 4. Perceptron “Relatives”: SVM and MLP

### 4.1 SVM vs Perceptron (both use linear hyperplanes, but with different objectives)

| Aspect            | Perceptron                                  | SVM (Soft-Margin)                                                                 |
|-------------------|---------------------------------------------|------------------------------------------------------------------------------------|
| Objective         | Reduce misclassifications (perceptron loss) | **Maximize margin** with regularization: $\tfrac12\|w\|^2 + C \sum \max(0, 1 - y f)$ |
| Margin control    | Not explicit                                | **Explicit** (hinge loss + $\|w\|^2$)                                         |
| Robustness        | Sensitive to noise/outliers                 | Typically **more robust** (margin principle)                                      |
| Solution support  | Uses all mistakes                           | Determined by **support vectors**                                                 |
| Kernel trick      | Possible but uncommon                       | Natural via dual/KKT                                                               |

**Bottom line**: Perceptron = “just separate”; SVM = “separate with the **widest safety buffer**.”

### 4.2 MLP vs Perceptron (nonlinear generalization)

| Aspect        | Perceptron (single layer)          | MLP (multi-layer perceptron)                                      |
|---------------|-------------------------------------|--------------------------------------------------------------------|
| Structure     | Linear hyperplane                   | Stacked linear transforms + **nonlinear activations**              |
| Capacity      | Linear decision boundary only       | Nonlinear; **universal approximation** (with enough width/depth)   |
| Training      | Error-driven updates                | Backprop + optimizers (SGD/Adam), regularization, normalization    |
| Output        | Hard label (±1)                     | Soft probabilities (sigmoid/softmax)                               |

**Classic example**: XOR is not linearly separable → Perceptron fails; a small MLP with nonlinear activations solves it easily.

### 4.3 Practical selection
- **Linear/near-linear, small data, fast baseline/online** → Perceptron (try **Averaged** / **Pocket**).  
- **Need robustness & margins or kernels** → SVM (linear or kernel).  
- **Nonlinear interactions & adequate data/compute** → MLP (or structure-aware nets such as CNN/RNN/Transformer).

---

## Appendix: Quick CLI sanity check
```bash
# Create venv and install numpy
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy -q
```
```python
# Save the classes above into perceptron_impl.py, then run:
import numpy as np
from perceptron_impl import Perceptron, PocketPerceptron

X = np.array([[2,3],[4,5],[1,0],[2,1]], dtype=float)
y = np.array([1,1,-1,-1])

print("== Averaged Perceptron ==")
clf = Perceptron(lr=1.0, max_iter=10, averaged=True, random_state=42).fit(X, y)
print("w,b:", clf.w_, clf.b_, "pred:", clf.predict(X))

print("\n== Pocket Perceptron ==")
X2 = np.array([[2,3],[4,5],[1,0],[2,1],[2.2,1.1],[1.8,3.2],[3.5,4.1]], dtype=float)
y2 = np.array([ 1,   1,   -1,   -1,    -1,        1,        1])
pp = PocketPerceptron(lr=1.0, max_iter=50, random_state=0).fit(X2, y2)
print("best_errors:", pp.best_errors_, "w*,b*:", pp.w_best_, pp.b_best_)
```