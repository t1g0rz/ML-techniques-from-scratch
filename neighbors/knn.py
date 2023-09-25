from typing import Union

import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        assert metric in {'euclidean', 'chebyshev', 'manhattan', 'cosine'}, \
            f"unknown distance metric specified `{metric}`"
        assert weight in {'uniform', 'rank', 'distance'}, f"unknown weight parameter `{weight}`"
        self.k = k
        self.train_size = (float('nan'), float('nan'))
        self.metric = metric
        self.weight = weight

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"
    
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(k={self.k}, metric='{self.metric}', weight='{self.weight}')"
    
    def _euclidean_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train)**2).sum(axis=1).pow(0.5)
    
    def _chebyshev_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train).abs()).max(axis=1)
        
    def _manhattan_distance(self, row: pd.Series) -> pd.Series:
        return ((row - self._X_train).abs()).sum(axis=1)
    
    def _cosine_distance(self, row: pd.Series) -> pd.Series:
        return (1 - (row * self._X_train).sum(axis=1) / 
                ((row.pow(2).sum()**0.5) * (self._X_train.pow(2).sum(axis=1).pow(0.5))))
    
    def _get_distance(self, row: pd.Series, return_prob: bool = False) -> Union[int, float]:
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
            if not return_prob:
                # in case of few modes get the max class number
                return self._y_train.loc[neighbors].mode().max()

            neighbors = self._y_train.loc[neighbors]
            # return a probability of class `1`
            return neighbors.sum() / len(neighbors)
        else:
            if self.weight == 'distance':
                w_distance = 1 / neighbors.nsmallest(self.k)
            elif self.weight == 'rank':
                w_distance = 1 / neighbors.rank().nsmallest(self.k)
            else:
                raise RuntimeError(f"Unknown weight: {self.weight}, please choose one of ['uniform', 'distance', "
                                   f"'rank']")
            w_distance.index = self._y_train.loc[w_distance.index]
            w_distance = (w_distance.groupby(level=0).sum() / w_distance.sum())
            if not return_prob:
                return w_distance.idxmax()
            else:
                # return a probability of class `1`
                return w_distance.loc[1] if 1 in w_distance else 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape
        self._X_train = X
        self._y_train = y

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(self._get_distance, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(self._get_distance, axis=1, return_prob=True)
