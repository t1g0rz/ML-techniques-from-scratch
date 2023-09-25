import pandas as pd
import numpy as np


class MyKNNReg:

    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        assert metric in {'euclidean', 'chebyshev', 'manhattan', 'cosine'}, \
            f"unknown distance metric specified `{metric}`"
        self.k = k
        self.train_size = (None, None)
        self._X_train = None
        self._y_train = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"

    def _euclidean_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train)**2).sum(axis=1).pow(0.5)

    def _chebyshev_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train).abs()).max(axis=1)

    def _manhattan_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train).abs()).sum(axis=1)

    def _cosine_distance(self, row: pd.Series) -> pd.Series:
        return (1 - (row * self._X_train).sum(axis=1) /
                ((row.pow(2).sum() ** 0.5) * (self._X_train.pow(2).sum(axis=1).pow(0.5))))

    def _get_distance(self, row: pd.Series):

        if self.metric == 'euclidean':
            distance = self._euclidean_distance
        elif self.metric == 'chebyshev':
            distance = self._chebyshev_distance
        elif self.metric == 'manhattan':
            distance = self._manhattan_distance
        elif self.metric == 'cosine':
            distance = self._cosine_distance
        else:
            raise RuntimeError(f"Unknown metric: {self.metric}, please choose one of ['euclidean', 'chebyshev', "
                               f"'manhattan', 'cosine']")

        neighbors = distance(row)

        if self.weight == 'uniform':
            neighbors = neighbors.nsmallest(self.k).index
            return self._y_train.loc[neighbors].mean()
        else:
            if self.weight == 'distance':
                w_distance = 1 / neighbors.nsmallest(self.k)
            elif self.weight == 'rank':
                w_distance = 1 / neighbors.rank().nsmallest(self.k)
            else:
                raise RuntimeError(f"Unknown weight: {self.weight}, please choose one of ['uniform', 'distance', "
                                   f"'rank']")
            return (w_distance / w_distance.sum() * self._y_train.loc[w_distance.index]).sum()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.train_size = X_train.shape
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X: pd.DataFrame):
        return X.apply(self._get_distance, axis=1)
