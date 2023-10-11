from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import module_a as m_a
import module_b as m_b

#a) polynom
np.random.seed(0)
X = np.sort(5 * np.random.rand(90, 1), axis=0)
y = m_a.polynom_3(X.ravel()) + np.random.normal(0, 6, X.shape[0])

plt.figure(figsize=(8, 6))
plt.scatter(X, y, label="train", c = "b", linewidths = 0.1)
plt.grid(True)
plt.legend()

model = RandomForestRegressor(n_estimators=1000, random_state=0, max_depth = 4)
model.fit(X, y)

mse = mean_squared_error(y, model.predict(X))
print("Mean Squared Error (MSE):", mse)


X_test = np.arange(0.0, 5.0, 0.1)[:, np.newaxis]
y_test = m_a.polynom_3(X_test.ravel())+ np.random.normal(0, 6, X_test.shape[0])

y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, label="test", c = "g", linewidths = 0.1)
plt.plot(X_test, y_pred, label="predict", c = "r")
plt.grid(True)
plt.legend()
plt.title("Random Forest Regression")

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)


#b) hyperbola

X_h = np.sort(5 * np.random.rand(90, 1), axis=0)
y_h = hyperbola(X_h.ravel()) + np.random.normal(0, 1, X_h.shape[0])
plt.figure(figsize=(8, 6))
plt.scatter(X_h, y_h, label="train", c = "b", linewidths = 0.1)
plt.grid(True)
plt.legend()

model = RandomForestRegressor(n_estimators=1000, random_state=0, max_depth = 3)
model.fit(X_h, y_h)

mse_2_train = mean_squared_error(y, model.predict(X_h))
print("Mean Squared Error (MSE):", mse_2_train)

X_h_test = np.arange(0.1, 6.0, 0.1)[:, np.newaxis]
y_h_test = m_b.hyperbola(X_h_test.ravel()) + np.random.normal(0, 1, X_h_test.shape[0])
y_h_pred = model.predict(X_h_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_h_test, y_h_test, label="test", c = "g", linewidths = 0.1)
plt.plot(X_h_test, y_h_pred, label="predict", c = "r")
plt.grid(True)
plt.legend()
plt.title("Random Forest Regression hyperbola")

mse_test = mean_squared_error(y_h_test, y_h_pred)
print("Mean Squared Error (MSE):", mse_test)
