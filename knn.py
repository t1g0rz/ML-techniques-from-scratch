from typing import Union

import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.train_size = (float('nan'), float('nan'))

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"
    
    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}(k={self.k})"

    def _euclidean_distance(self, row: pd.Series, return_prob: bool = False) -> Union[int, float]:
        neighbors = ((row - self._X_train)**2).sum(axis=1).pow(0.5).nsmallest(self.k).index
        if not return_prob:
            # in case of few modes
            return self._y_train.loc[neighbors].mode().max()

        neighbors = self._y_train.loc[neighbors]
        # return a probability of class `1`
        return neighbors.sum() / len(neighbors)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.train_size = X.shape
        self._X_train = X
        self._y_train = y

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(self._euclidean_distance, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return X.apply(self._euclidean_distance, axis=1, return_prob=True)
