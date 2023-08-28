import math
import random
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


class LinearRegression:

    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1, metric: Optional[str] = None,
                 reg : Optional[str] = None, l1_coef: float = 0., l2_coef: float = 0.,
                 sgd_sample: Union[int, float, None] = None, random_state: Optional[int] = 1) -> None:
        """
        Initialize the LinearRegression class.

        Parameters:
        - n_iter (int): Number of iterations for gradient descent.
        - learning_rate (Union[float, Callable]): Learning rate for gradient descent. Can be a fixed value or a function,
        which changes its value depending on the gradient descent step, e.g. `lambda step: 0.5 * (0.85 ** step)`
        - metric (Optional[str]): Metric for evaluating the model (choices: 'mae', 'mse', 'rmse', 'mape', 'r2').
        - reg (Optional[str]): Regularization type (choices: 'l1', 'l2', 'elasticnet').
        - l1_coef (float): Coefficient for L1 regularization.
        - l2_coef (float): Coefficient for L2 regularization.
        - sgd_sample (Union[int, float, None]): Sample size for stochastic gradient descent. Can be an absolute number, 
        a fraction of the total, or None.
        - random_state (int): Random seed for reproducibility. If set to None, each iteration will be chosen random 
        sample for stochastic grafient descent
        """
        assert metric in {'mae', 'mse', 'rmse', 'mape', 'r2', None}, f"unexpected metric {metric}"
        assert reg in {'l1', 'l2', 'elasticnet', None}, f"unexpected regularization parameter {reg}"
        assert 0 <= l1_coef <= 1. and 0 <= l2_coef <= 1., f"l1_coef and l2_coef have to be between 0. and 1."
        self.metric = metric
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.best_metric = float('inf')
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        if reg == 'l1':
            self.l1_coef = l1_coef
            self.l2_coef = 0.
        elif reg == 'l2':
            self.l2_coef = l2_coef
            self.l1_coef = 0.
        elif reg == 'elasticnet':
            self.l1_coef = l1_coef
            self.l2_coef = l2_coef
        else:
            self.l1_coef = 0.
            self.l2_coef = 0.
    
    def __str__(self) -> str:
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __repr__(self) -> str:
        return self.__str__()
    
    def _mse(self, y_p: np.ndarray, y: np.ndarray) -> float:
        """Return mean sqared error (MSE) with regularization since this metric used as a loss function
        https://en.wikipedia.org/wiki/Mean_squared_error

        Parameters:
        - y_p (np.ndarray): _description_
        - y (np.ndarray): _description_

        Returns:
        - float: Mean sqared error
        """
        return ((y_p - y)**2).sum() / len(y) + self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * (self.weights**2).sum()

    def _mae(self, y_p: np.ndarray, y: np.ndarray) -> float:
        return (y_p - y).abs().sum() / len(y)

    def _rmse(self, y_p: np.ndarray, y: np.ndarray) -> float:
        return math.sqrt(self._mse(y_p, y))

    def _mape(self, y_p: np.ndarray, y: np.ndarray) -> float:
        return ((y - y_p) / y).abs().sum() * 100 / len(y)

    def _r2(self, y_p: np.ndarray, y: np.ndarray) -> float:
        return 1 - (((y - y_p)**2).sum() / ((y - y.mean())**2).sum())

    def _get_metric(self, **kwargs) -> Optional[float]:
        """
        Compute the chosen metric for the model.

        Parameters:
        - y_p (np.ndarray): Predicted values.
        - y (np.ndarray): True values.
        
        Returns:
        - float: Computed metric value.
        """
        if not self.metric:
            return
        elif self.metric == 'mae':
            return self._mae(**kwargs)
        elif self.metric == 'mse':
            return self._mse(**kwargs)
        elif self.metric == 'rmse':
            return self._rmse(**kwargs)
        elif self.metric == 'mape':
            return self._mape(**kwargs)
        elif self.metric == 'r2':
            return self._r2(**kwargs)
        else:
            raise Exception(f"Unknown metric '{self.metric}'!")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], verbose: int = 0) -> None:
        """
        Train the linear regression model using gradient descent.

        Parameters:
        - X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
        - y (Union[pd.Series, np.ndarray]): Target values.
        - verbose (Union[int, bool]): If int, print progress every 'verbose' steps. 
        If verbose = int, print progress every ith step. If False (or 0), no output.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        X = np.insert(X, 0, 1, axis=1)
        self.weights = np.ones(X.shape[1])
        if self.random_state:
            random.seed(self.random_state)

        loss = self._mse
        y_hat = (X @ self.weights)
        
        if self.metric:
            self.best_metric = self._get_metric(y_p=y_hat, y=y)
        if verbose:
            verbose_string = f"start   | loss: {loss(y_hat, y)}"
            if self.metric:
                verbose_string += f" | {self.metric}: {self.best_metric}"
            print(verbose_string)
        # SGD
        if isinstance(self.sgd_sample, float):
            k = min(round(X.shape[0] * self.sgd_sample), X.shape[0])
        elif isinstance(self.sgd_sample, int):
            k = min(self.sgd_sample, X.shape[0])
        else:
            k = X.shape[0]

        for i in range(1, self.n_iter + 1):
            if X.shape[0] == k:
                X_smpl, y_hat_sampl, y_sampl = X, y_hat, y
            else:
                sample_rows_idx = random.sample(range(X.shape[0]), k)
                X_smpl, y_hat_sampl, y_sampl = X[sample_rows_idx], y_hat[sample_rows_idx], y[sample_rows_idx]

            gradient = (2 / k * (y_hat_sampl - y_sampl) @ X_smpl) + self.l1_coef * np.sign(self.weights)\
                  + self.l2_coef * 2 * self.weights
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate
            self.weights = self.weights - lr * gradient
            y_hat = (X @ self.weights)

            if self.metric:
                self.best_metric = self._get_metric(y_p=y_hat, y=y)
            if verbose and (i) % verbose == 0:
                verbose_string = f"{i:<7} | loss: {loss(y_hat, y)}"
                if self.metric:
                    verbose_string += f" | {self.metric}: {self.best_metric}"
                print(verbose_string)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for the given feature matrix.

        Parameters:
        - X (pd.DataFrame): Feature matrix.
        
        Returns:
        - np.ndarray: Predicted values.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.insert(X, 0, 1, axis=1)
        y_hat = (X @ self.weights)
        return y_hat

    def get_coef(self) -> np.ndarray:
        """
        Retrieve the coefficients of the trained model except of w0

        Returns:
        - np.ndarray: Model coefficients.
        """
        return self.weights[1:]
    
    def get_best_score(self) -> float:
        """
        Get the best score achieved during training based on the chosen metric.

        Returns:
        - float: Metric score calculated on the last step.
        """
        return self.best_metric
    