import random
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self, n_iter: int = 10, learning_rate: Union[float, Callable] = 0.1, metric: Optional[str] = None,
                 reg : Optional[str] = None, l1_coef: float = 0., l2_coef: float = 0.,
                 sgd_sample: Union[int, float, None] = None, random_state: Optional[int] = None) -> None:
        """
        Initialize the LogicticRegresion class.

        Parameters:
            `n_iter` (int, optional): Number of iterations for optimization. Defaults to 10.
            `learning_rate` (Union[float, Callable], optional): Learning rate or a function to calculate the learning 
            rate. Defaults to 0.1.
            metric (Optional[str], optional): Metric for evaluation ('accuracy', 'precision', 'recall', 'f1', 
            'roc_auc'). Defaults to None.
            `reg` (Optional[str], optional): Type of regularization ('l1', 'l2', 'elasticnet'). Defaults to None.
            `l1_coef` (float, optional): Coefficient for L1 regularization. Valid if reg is 'l1' or 'elasticnet'. 
            Defaults to 0.
            `l2_coef` (float, optional): Coefficient for L2 regularization. Valid if reg is 'l2' or 'elasticnet'.\
            Defaults to 0.
            `sgd_sample` (Union[int, float, None], optional): Sample size or fraction for stochastic gradient descent.
            Defaults to None.
            `random_state` (Optional[int], optional): Seed for random number generator. Defaults to None.

        Raises:
            AssertionError: If provided metric or regularization type is not valid.
            AssertionError: If l1_coef or l2_coef is not between 0 and 1.
        """

        assert metric in {'accuracy', 'precision', 'recall', 'f1', 'roc_auc', None}
        assert reg in {'l1', 'l2', 'elasticnet', None}, f"unexpected regularization parameter {reg}"
        assert 0 <= l1_coef <= 1. and 0 <= l2_coef <= 1., f"l1_coef and l2_coef have to be between 0. and 1."
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.metric = metric
        self.last_metric = float('inf')
        self.random_state = random_state
        self.sgd_sample = sgd_sample
        self.reg = reg
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
    
    def __repr__(self) -> str:
        class_name = type(self).__name__
        if self.reg:
            reg = f"'{self.reg}'"
        else:
            reg = self.reg
        if self.metric:
            metric = f"'{self.metric}'"
        else:
            metric = self.metric
        return f"{class_name}(n_iter={self.n_iter}, learning_rate={self.learning_rate}, metric={metric}, "\
               f"reg={reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}, "\
               f"sgd_sample={self.sgd_sample}, random_state={self.random_state})"

    def _confusion_mtrx(self, y: np.ndarray, y_hat: np.ndarray) -> (int, int, int, int):
        """
        Generate a confusion matrix for binary classification.

        Parameters:
            y (np.ndarray): True labels.
            y_hat (np.ndarray): Predicted probabilities.

        Returns:
            tuple: A tuple containing True Positive, True Negative, False Positive, and False Negative counts.
        """
        v = np.array([y, np.round(y_hat)])
        true_positive = sum((v[0] == 1) & (v[1] == 1))
        true_negative = sum((v[0] == 0) & (v[1] == 0))
        false_positive = sum((v[0] == 0) & (v[1] == 1))   # type 1 error
        false_negative = sum((v[0] == 1) & (v[1] == 0))   # type 2 error
        return true_positive, true_negative, false_positive, false_negative

    def _accuracy(self, *, tp: int, tn: int, fp: int, fn: int) -> float:
        return (tp + tn) / (tp + tn + fp + fn)

    def _precision(self, *, tp: int, fp: int) -> float:
        return tp / (tp + fp)
    
    def _recall(self, *, tp: int, fn: int) -> float:
        return tp / (tp + fn)

    def _f_measure(self, *, tp: int, tn: int, fp: int, fn: int, beta: float) -> float:
        precision = self._precision(tp=tp, fp=fp)
        recall = self._recall(tp=tp, fn=fn)
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    def _roc_auc(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        p = sum(y == 1)
        n = y.shape[0] - p
        y_hat = np.round(y_hat, 10)
        v = pd.concat([pd.Series(y_hat, name='prob'), pd.Series(y, name='class')], axis=1)
        v = v.sort_values(['prob', 'class'], ascending=[False, True])
        v['cum_count'] = v['class'].cumsum()
        # find those spots where should be 0.5
        w = v.groupby('prob')['class'].nunique()
        return (v[v['class'] == 0].cum_count.sum() + w[w == 2].shape[0] * 0.5)  / (p * n) 

    def _get_metric(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the specified evaluation metric.

        Parameters:
            y (np.ndarray): True labels.
            y_hat (np.ndarray): Predicted probabilities.

        Returns:
            float: Computed metric value.
        """

        if self.metric != 'roc_auc':
            tp, tn, fp, fn = self._confusion_mtrx(y, y_hat)
        if self.metric == 'accuracy':
            return self._accuracy(tp=tp, tn=tn, fp=fp, fn=fn)
        elif self.metric == 'f1':
            return self._f_measure(tp=tp, tn=tn, fp=fp, fn=fn, beta=1.)
        elif self.metric == 'recall':
            return self._recall(tp=tp, fn=fn)
        elif self.metric == 'precision':
            return self._precision(tp=tp, fp=fp)
        elif self.metric == 'roc_auc':
            return self._roc_auc(y, y_hat)
        else:
            raise Exception(f"Not realized metric: {self.metric}")

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], verbose: Optional[int] = None):
        """
        Fit the logistic regression model using (stochastic) gradient descent.

        Parameters:
            X (Union[np.ndarray, pd.DataFrame]): Feature matrix.
            y (np.ndarray): True labels.
            verbose (int, optional): Interval for printing training progress. If set to 0, no progress will be printed. 
            Defaults to 0.

        Raises:
            AssertionError: If input shapes are inconsistent.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if self.random_state:
            random.seed(self.random_state)
        X = np.insert(X, 0, 1, axis=1)
        n = X.shape[0]
        eps = 1e-15
        self.weights = np.ones(X.shape[1])

        y_hat = 1 / (1 + np.e**-(X @ self.weights))
        if self.metric:
            self.best_metric = self._get_metric(y, y_hat)
        if verbose:
            verbose_string = f"start     | loss: "\
                f"{-1 / n * (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).sum()}"
            if self.metric:
                verbose_string += f" | {self.metric}: {self.best_metric}"
            print(verbose_string)

        # Stochastic gradient descent
        if isinstance(self.sgd_sample, float):
            k = min(round(X.shape[0] * self.sgd_sample), X.shape[0])
        elif isinstance(self.sgd_sample, int):
            k = min(self.sgd_sample, X.shape[0])
        else:
            k = X.shape[0]

        for i in range(1, self.n_iter+1):

            if X.shape[0] == k:
                X_smpl, y_hat_sampl, y_sampl = X, y_hat, y
            else:
                sample_rows_idx = random.sample(range(X.shape[0]), k)
                X_smpl, y_hat_sampl, y_sampl = X[sample_rows_idx], y_hat[sample_rows_idx], y[sample_rows_idx]
            
            log_loss = -1 / k * (y_sampl * np.log(y_hat_sampl + eps) + (1 - y_sampl) * np.log(1 - y_hat_sampl + eps)).sum() \
                + self.l1_coef * np.abs(self.weights).sum() + self.l2_coef * (self.weights**2).sum()

            gradient = 1 / k * (y_hat_sampl - y_sampl) @ X_smpl + self.l1_coef * np.sign(self.weights)\
                  + self.l2_coef * 2 * self.weights
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate
            self.weights = self.weights - lr * gradient
            y_hat = 1 / (1 + np.e**-(X @ self.weights))
            if self.metric:
                self.last_metric = self._get_metric(y, y_hat)
            if verbose and (i) % verbose == 0:
                verbose_string = f"{i:<7}   | loss: {log_loss}"
                if self.metric:
                    verbose_string += f" | {self.metric}: {self.last_metric}"
                print(verbose_string)
    
    def get_coef(self) -> np.ndarray:
        """
        Retrieve the coefficients of the fitted logistic regression model.

        Returns:
            np.ndarray: Coefficients of the model.
        """
        return self.weights[1:]

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
            X (Union[np.ndarray, pd.DataFrame]): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        y_hat = self.predict_probability(X)
        return np.round(y_hat).astype(int)

    def predict_probability(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters:
            X (Union[np.ndarray, pd.DataFrame]): Feature matrix.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.insert(X, 0, 1, axis=1)
        y_hat = 1 / (1 + np.e**-(X @ self.weights))
        return y_hat

    def get_best_score(self) -> float:
        """
        Retrieve the last score achieved during training based on the specified metric.

        Returns:
            float: Last score achieved.
        """
        return self.last_metric
