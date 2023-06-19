import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Read the data from the file
data = []
with open("coordinates.txt", "r") as f:
    for line in f:
        x, y, _ = line.strip().split(",")
        # Parse x and y as floats
        x_val = float(x.split("=")[1].strip())
        y_val = float(y.split("=")[1].strip())
        data.append((x_val, y_val))

df = pd.DataFrame(data, columns=["x", "y"])

# Fit a polynomial regression model to the data
X = df["x"].values.reshape(-1, 1)
y = df["y"].values

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression().fit(X_poly, y)

# Generate the regression equation
coefs = model.coef_
intercept = model.intercept_

equation = f"y = {intercept} + "
for i, coef in enumerate(coefs[1:]):
    equation += f"{coef}*x^{i + 1} + "
equation = equation[:-3]  # remove the last '+ '

print("Regression equation:", equation)

# Plot the data and the regression curve
plt.scatter(X, y, color="blue", label="Data")

X_plot = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
y_plot = model.predict(poly.transform(X_plot))

plt.plot(X_plot, y_plot, color="red", label="Regression")
plt.legend()
plt.show()
