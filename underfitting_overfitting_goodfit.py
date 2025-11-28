import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create sample data (non-linear curve)
np.random.seed(0)
X = np.linspace(0, 10, 20)
y = np.sin(X) + np.random.normal(0, 0.2, len(X))  # curved data

X = X.reshape(-1, 1)

# 1. Underfitting: Linear Regression (degree 1)
model_under = LinearRegression()
model_under.fit(X, y)

# 2. Good Fit: Polynomial Degree 3
model_good =make_pipeline(PolynomialFeatures(3), LinearRegression())
model_good.fit(X, y)

# 3. Overfitting: Polynomial Degree 10
model_over =make_pipeline(PolynomialFeatures(10), LinearRegression())
model_over.fit(X, y)

# Prediction range
X_test = np.linspace(0, 10, 200).reshape(-1, 1)

y_under = model_under.predict(X_test)
y_good  = model_good.predict(X_test)
y_over  = model_over.predict(X_test)


# ---------- PLOT ----------
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(X_test, y_under, label='Underfitting (Linear)', linewidth=2)
plt.plot(X_test, y_good, label='Good Fit (Degree 3)', linewidth=2)
plt.plot(X_test, y_over, label='Overfitting (Degree 10)', linewidth=2)
plt.title("Underfitting vs Good Fit vs Overfitting")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
