from typing import Union, Optional, Callable

import numpy as np
import pandas as pd
import random


class LogicticRegresion:

    def __init__(self, n_iter: int = 10, learning_rate: Union[float, Callable] = 0.1, metric: Optional[str] = None,
                 reg : Optional[str] = None, l1_coef: float = 0., l2_coef: float = 0.,
                 sgd_sample: Union[int, float, None] = None, random_state: Optional[int] = 42) -> None:
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
        return f"LogicticRegresion class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def _confusion_mtrx(self, y: np.ndarray, y_hat: np.ndarray) -> (int, int, int, int):
        """Generate confusin matrix. Since we have binary classes output should have shape (2, 4)

        Args:
            y (np.ndarray): _description_
            y_hat (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
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
            verbose_string = f"start    | loss: "\
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
        return self.weights[1:]

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        y_hat = self.predict_proba(X)
        return np.round(y_hat).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.insert(X, 0, 1, axis=1)
        y_hat = 1 / (1 + np.e**-(X @ self.weights))
        return y_hat

    def get_best_score(self) -> float:
        return self.last_metric
