from typing import Union, Optional

import numpy as np
import pandas as pd



class MyLogRegresion:

    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        
    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], verbose: Optional[int] = None):

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.insert(X, 0, 1, axis=1)
        n = X.shape[0]
        eps = 1e-15
        self.weights = np.ones(X.shape[1])

        for i in range(self.n_iter):
            y_hat = 1 / (1 + np.e**-(X @ self.weights))
            log_loss = -1 / n * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).sum()

            gradient = 1 / n * (y_hat - y) @ X 
            self.weights = self.weights - self.learning_rate * gradient

    
    def get_coef(self) -> np.ndarray:
        return self.weights[1:]
    

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        pass

    def predict_proba(self, ) -> np.ndarray:
        pass
