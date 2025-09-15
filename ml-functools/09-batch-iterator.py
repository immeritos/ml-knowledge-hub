import numpy as np
from typing import Iterator, Tuple, Optional

def batch_iterator(
    X: np.ndarray, 
    y: Optional[np.ndarray] = None, 
    batch_size: int =64
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    n_samples = X.shape[0]
    
    for start in range(0, n_samples, batch_size):
        end = min(start+batch_size, n_samples)
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end], None