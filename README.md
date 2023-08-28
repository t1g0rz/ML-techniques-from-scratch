# ML Techniques From Scratch

## Purpose
_By developing ML techniques, we aim to translate the theoretical knowledge available on the Internet into a practical tool._


## LinearRegression
`LinearRegression` is a basic implementation of a linear regression model with options for regularization and different evaluation metrics.  


### Features:
- Gradient Descent Training: Uses gradient descent to optimize the model parameters.  
- Multiple Evaluation Metrics: Supports various evaluation metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared.
- Regularization: Supports `L1 (Lasso)`, `L2 (Ridge)`, and `ElasticNet` regularization techniques.
- Stochastic Gradient Descent (SGD): Option to use a subset of data for each iteration of gradient descent.

### Parameters:
- `n_iter` (int): Number of iterations for gradient descent.
learning_rate (Union[float, Callable]): Learning rate for gradient descent. Can be a fixed value or a function.
- `metric` (Optional[str]): Metric for evaluating the model.
- `reg` (Optional[str]): Regularization type.
- `l1_coef` (float): Coefficient for L1 regularization.
- `l2_coef` (float): Coefficient for L2 regularization.
- `sgd_sample` (Union[int, float, None]): Sample size for stochastic gradient descent.
- `random_state` (int): Random seed for reproducibility.

### Usage:
Initialization:
```python
model = MyLineReg(n_iter=100, learning_rate=0.1, metric='mse', reg='elasticnet', l1_coef=0.1, l2_coef=0.1, sgd_sample=0.1)
```

Training:
```python
model.fit(X_train, y_train, verbose=10)
```

Predictions:
```python
predictions = model.predict(X_test)
```

Retrieving Coefficients except `w0`:
```python
coefficients = model.get_coef()
```

Retrieving Last Metric Score:
```python
best_score = model.get_best_score()
```
