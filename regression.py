import pandas as pd  # For handling datasets
import numpy as np  # For numerical operations
from sklearn.linear_model import LinearRegression  # To perform linear regression
from sklearn.preprocessing import (
    PolynomialFeatures,
)  # To transform input for polynomial regression
import matplotlib.pyplot as plt  # To plot data

# Create an empty list to store data
data = []

# Open the coordinates file and read its content
with open("coordinates.txt", "r") as f:
    # Iterate through each line in the file
    for line in f:
        # Split the line into x, y, and probability using the comma separator
        x, y, _ = line.strip().split(",")
        # For x and y, split again using the equal sign and convert to float
        x_val = float(x.split("=")[1].strip())
        y_val = float(y.split("=")[1].strip())
        # Add the (x, y) pair to our data list
        data.append((x_val, y_val))

# Convert the list into a DataFrame for easier manipulation
df = pd.DataFrame(data, columns=["x", "y"])

# Convert x and y columns to numpy arrays and reshape x to a 2D array because
# sklearn requires it in this format
X = df["x"].values.reshape(-1, 1)
y = df["y"].values

# Create a PolynomialFeatures object with degree 4.
# This will transform our input from X to [1, X, X^2, X^3, X^4]
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Create a LinearRegression model and fit it to our transformed input and output
model = LinearRegression().fit(X_poly, y)

# Retrieve the coefficients and intercept of the polynomial
coefs = model.coef_
intercept = model.intercept_

# Construct a string for the regression equation
equation = f"y = {intercept} + "
for i, coef in enumerate(coefs[1:]):
    # Add each term to the equation
    equation += f"{coef}*x^{i + 1} + "
# Remove the last '+ ' from the equation
equation = equation[:-3]

print("Regression equation:", equation)

# Create a scatter plot of the original data
plt.scatter(X, y, color="blue", label="Data")

# Generate 1000 equally spaced X values from min(X) to max(X) for plotting the regression curve
X_plot = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
# Calculate the corresponding y values for the plot
y_plot = model.predict(poly.transform(X_plot))

# Plot the regression curve
plt.plot(X_plot, y_plot, color="red", label="Regression")

# Add a legend and display the plot
plt.legend()
plt.show()
