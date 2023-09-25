import pandas as pd
import numpy as np
from typing import Union, Optional


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
        self.max_leaves = max_leafs if max_leafs >= 2 else 2
        self.leaves_cnt = 0
        self._splits_cnt = 1  # imply that root should be in any case
        self.tree: Optional[TreeNode] = None

    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, "\
               f"max_leafs={self.max_leaves}"

    @staticmethod
    def _calculate_entropy(y: pd.Series) -> float:
        p = y.value_counts() / len(y)
        return -(p * p.apply(np.log2)).sum()

    def _get_best_split(self, X, y) -> Union[TreeNode, Leaf]:
        s0 = self._calculate_entropy(y)
        N = len(y)
        if len(y.value_counts()) == 1:
            self.leaves_cnt += 1
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
        if depth >= self.max_depth or self._splits_cnt + 1 == self.max_leaves:
            self.leaves_cnt += 2
            node.left = Leaf(y_l.mean())
            node.right = Leaf(y_r.mean())
        else:
            if len(y_l) < self.min_samples_split or self._splits_cnt + 1 == self.max_leaves:
                self.leaves_cnt += 1
                node.left = Leaf(y_l.mean())
            else:
                node.left = self._get_best_split(X_l, y_l)
            if isinstance(node.left, TreeNode):
                self._splits_cnt += 1
                self._build_tree(X_l, y_l, node.left, depth + 1)

            if len(y_r) < self.min_samples_split or self._splits_cnt + 1 == self.max_leaves:
                self.leaves_cnt += 1
                node.right = Leaf(y_r.mean())
            else:
                node.right = self._get_best_split(X_r, y_r)
            if isinstance(node.right, TreeNode):
                self._splits_cnt += 1
                self._build_tree(X_r, y_r, node.right, depth + 1)

    def _traverse_tree(self, row: pd.Series) -> float:
        node = self.tree
        while isinstance(node, TreeNode):
            if row[node.col_name] > node.split_value:
                node = node.right
            else:
                node = node.left

        return node.value

    def print_tree(self) -> None:
        print(self.tree)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.tree = self._get_best_split(X, y)
        self._build_tree(X, y, self.tree, depth=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        y_pred = self.predict_proba(X)
        flt = y_pred > 0.5
        y_pred.loc[flt] = 1
        y_pred.loc[~flt] = 0
        return y_pred.astype(int)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._traverse_tree, axis=1)


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'target']
    X, y = df.iloc[:, :4], df['target']
    X_train = X.sample(int(len(X) * 0.8), random_state=41)
    y_train = y.loc[X_train.index]
    flt = ~X.index.isin(X_train.index)
    y_test = y.loc[flt]
    X_test = X.loc[flt]
    t = MyTreeClf(max_depth=5, min_samples_split=5, max_leafs=20)
    t.fit(X, y)
    t.print_tree()

    print("Prediction quality\n", pd.concat([t.predict(X_test).rename('pred'), y_test], axis=1).corr())
