#Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2, 4, 5, 4, 5])

# Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression (EDA)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
