# main.py
print("Training simple ML model...")

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=5)
model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")
