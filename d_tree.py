import pandas as pd
import numpy as np
from typing import Union, Optional
# TODO: leafs -> leaves
# TODO: add leaves count


class Leaf:
    def __init__(self, value: float) -> None:
        self.value = value

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}({self.value})"

    def __str__(self):
        return self.__repr__()


class TreeNode:
    def __init__(self, col_name: str, split_value: float, ig: float,
                 left: Union["TreeNode", Leaf, None] = None, right: Union["TreeNode", Leaf, None] = None) -> None:
        self.col_name = col_name
        self.split_value = split_value
        self.ig = ig
        self.left = left
        self.right = right

    def __repr__(self):
        return (f"TreeNode({self.col_name}, {self.split_value}, {self.ig}, "
                f"{self.left.__repr__()}, {self.right.__repr__()})")

    def __str__(self):
        result = ""
        stack = [(self, 0, "root")]
        while stack:
            node, indent, side = stack.pop()
            if isinstance(node, TreeNode):
                result += f"{' ' * indent}{node.col_name} > {node.split_value:.8f}  |  IG: {node.ig:.8f}\n"
                stack.append((node.right, indent + 2, 'right'))
                stack.append((node.left, indent + 2, 'left'))
            elif isinstance(node, Leaf):
                result += f"{' ' * indent}leaf_{side} = {node.value}\n"
            else:
                result += f"{' ' * indent}<UNKNOWN NODE>\n"
        return result


class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree: Optional[TreeNode] = None

    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "\
               f"max_leafs={self.max_leafs}"

    @staticmethod
    def _calculate_entropy(y: pd.Series) -> float:
        p = y.value_counts() / len(y)
        return -(p * p.apply(np.log2)).sum()

    def _get_best_split(self, X, y) -> Union[TreeNode, Leaf]:
        s0 = self._calculate_entropy(y)
        N = len(y)
        if len(y.value_counts()) == 1:
            return Leaf(float(y.iloc[0]))
        col_name, split_value, ig = '', float('-inf'), 0

        for col in X.columns:
            tmp = pd.Series(X[col].sort_values().unique()).rolling(2).mean()
            for split in tmp.iloc[1:]:
                flt = X[col] <= split
                part1, part2 = y.loc[flt], y.loc[~flt]
                s1, s2 = self._calculate_entropy(part1), self._calculate_entropy(part2)
                new_ig = s0 - s1 * len(part1) / N - s2 * len(part2) / N
                if new_ig > ig:
                    col_name, split_value, ig = col, split, new_ig

        return TreeNode(col_name, split_value, ig)

    def _build_tree(self, X, y, node, depth) -> None:
        flt = X[node.col_name] <= node.split_value
        X_l, y_l = X[flt], y[flt]
        X_r, y_r = X[~flt], y[~flt]
        if depth >= self.max_depth:
            node.left = Leaf(y_l.sum() / len(y_l))
            node.right = Leaf(y_r.sum() / len(y_r))
        else:
            node.left = self._get_best_split(X_l, y_l)
            if isinstance(node.left, TreeNode):
                self._build_tree(X_l, y_l, node.left, depth + 1)
            node.right = self._get_best_split(X_r, y_r)
            if isinstance(node.right, TreeNode):
                self._build_tree(X_r, y_r, node.right, depth + 1)

    def print_tree(self) -> None:
        print(self.tree)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.tree = self._get_best_split(X, y)
        self._build_tree(X, y, self.tree, depth=1)
