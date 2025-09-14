import numpy as np

def linear_regression_gradient(X, y, alpha=0.01, max_iter = 1000, tol=1e-6, add_intercept= True):
    """
    Linear Regression using Gradient Descent with early stopping.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    alpha : float
        Learning rate
    iterations : int
        Maximum number of iterations
    tol : float
        Tolerance for early stopping (stop if loss change < tol)
    add_intercept : bool
        Whether to add an intercept column of ones
    
    Returns
    -------
    intercept : float
    coef : np.ndarray
    losses : list
        Loss values during training
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    m, n = X.shape
    
    if add_intercept:
        ones = np.ones((m, 1))
        X = np.hstack([ones, X])
        n += 1

    # Initialize theta        
    theta = np.zeros((n, 1))
    
    losses = []
    prev_loss = float("inf")
    
    for _ in range(max_iter):
        # Prediction
        predictions = X @ theta
        # Error vector
        errors = predictions - y
        # Gradient
        gradient = (X.T @ errors) / m
        # Update
        theta -= alpha * gradient
        
        # Compute and store MSE loss        
        loss = (errors.T @ errors) / (2*m)
        losses.append(float(loss))
        
        if abs(prev_loss - loss) < tol:
            print(f"Converged early: loss change < {tol}")
            break
        prev_loss = loss
        
    intercept = float(theta[0, 0]) if add_intercept else 0.0
    coef = theta[1:, 0] if add_intercept else theta[:, 0]
    
    return intercept, coef, losses