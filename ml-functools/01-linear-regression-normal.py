import numpy as np

# ===========================================================
# Linear Regression via Normal Equation
# ===========================================================

def linear_regression_normal_equation(X: np.ndarray, y: np.ndarray, add_intercept=True):
    """
    Solve linear regression using the Normal Equation:
        theta = (X^T X)^(-1) X^T y
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    add_intercept : bool
        If True, automatically add a column of ones for the intercept
    
    Returns
    -------
    intercept : float
        The bias term (intercept)
    coef : np.ndarray
        The coefficients for each feature
    """

    # Ensure X and y are numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)  
    # Why reshape(-1,1)? → y must be a column vector (n,1) for matrix multiplication

    # Add a column of ones if we want the intercept
    if add_intercept:
        ones = np.ones((X.shape[0], 1))  # shape (n,1), not (n,)
        X = np.hstack([ones, X])

    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)  
    # Use pseudo-inverse (pinv) instead of inv for better numerical stability

    # Theta shape is (n_features+1, 1): 
    #   [intercept, coef1, coef2, ..., coef_p]^T
    intercept = float(theta[0, 0]) if add_intercept else 0.0
    coef = theta[1:, 0] if add_intercept else theta[:, 0]
    return intercept, coef.tolist()


# ===========================================================
# Linear Regression via QR Decomposition
# ===========================================================

def linear_regression_qr(X: np.ndarray, y: np.ndarray, add_intercept=True):
    """
    Solve linear regression using QR decomposition:
        X = QR
        => minimize ||y - X theta||_2
        => equivalent to solving R theta = Q^T y
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    add_intercept : bool
        Whether to add a column of ones for the intercept
    
    Returns
    -------
    intercept : float
        The bias term
    coef : np.ndarray
        The coefficients for each feature
    """

    # Convert to numpy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)  # make y a column vector

    # Add intercept column
    if add_intercept:
        ones = np.ones((X.shape[0], 1))  
        X = np.hstack([ones, X])

    # QR decomposition: X = Q R
    Q, R = np.linalg.qr(X)

    # Equivalent system: R theta = Q^T y
    Qt_y = Q.T.dot(y)

    # Solve for theta using back substitution
    theta = np.linalg.solve(R, Qt_y)

    # Extract intercept and coefficients
    intercept = float(theta[0, 0]) if add_intercept else 0.0
    coef = theta[1:, 0] if add_intercept else theta[:, 0]
    return intercept, coef.tolist()


# ===========================================================
# Example usage
# ===========================================================
if __name__ == "__main__":
    # Example dataset: 1D feature
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.2, 2.3, 2.9, 4.1, 5.1])

    # Normal Equation
    intercept_ne, coef_ne = linear_regression_normal_equation(X, y)
    print("Normal Equation -> Intercept:", intercept_ne, "Coef:", coef_ne)

    # QR Decomposition
    intercept_qr, coef_qr = linear_regression_qr(X, y)
    print("QR Decomposition -> Intercept:", intercept_qr, "Coef:", coef_qr)
