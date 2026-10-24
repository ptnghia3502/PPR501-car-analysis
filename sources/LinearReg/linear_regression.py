
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
import re

# Read CSV dataset
dataset = pd.read_csv("../Dataset/car-price.csv")

# Convert Vietnamese car prices to million VND
def convert_price(price):
    if pd.isna(price):
        return np.nan

    price = str(price).strip()

    ty = re.search(r"([\d.,]+)\s*Tỷ", price, re.IGNORECASE)
    trieu = re.search(r"([\d.,]+)\s*Triệu", price, re.IGNORECASE)

    ty_value = float(ty.group(1).replace(",", ".")) if ty else 0
    trieu_value = float(trieu.group(1).replace(",", ".")) if trieu else 0

    return ty_value * 1000 + trieu_value


# Prepare data
dataset["Price_Million"] = dataset["Price"].apply(convert_price)

# Use manufacturing year as input (X)
# Use car price as output (Y)
dataset["Năm sản xuất"] = pd.to_numeric(
    dataset["Năm sản xuất"], errors="coerce"
)

dataset = dataset.dropna(
    subset=["Năm sản xuất", "Price_Million"]
)

x = dataset[["Năm sản xuất"]].to_numpy()
y = dataset["Price_Million"].to_numpy()

# Train Linear Regression model
model = LinearRegression()
model.fit(x, y)

# Calculate RMSE and R squared
predictions = model.predict(x)

rmse = math.sqrt(mean_squared_error(y, predictions))

print("RMSE:", rmse, "million VND")
print("R squared value:", model.score(x, y))

# Regression equation
slope = model.coef_[0]
intercept = model.intercept_

intercept_sign = "+" if intercept >= 0 else "-"

equation = (
    f"Price = {slope:,.2f} * Year "
    f"{intercept_sign} {abs(intercept):,.2f}"
)

print("Regression equation:", equation)

# Visualize actual data and regression line
plt.scatter(
    x.flatten(),
    y,
    color="green",
    label="Actual car prices"
)

# Sort years to draw a clean regression line
sorted_indices = np.argsort(x.flatten())
x_sorted = x[sorted_indices]

plt.plot(
    x_sorted.flatten(),
    model.predict(x_sorted),
    color="black",
    label="Linear Regression"
)

plt.title("Car Price Prediction - Vietnam")
plt.xlabel("Manufacturing Year")
plt.ylabel("Price (Million VND)")
plt.legend()

plt.text(
    0.05,
    0.95,
    equation,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Predict car price for a specific manufacturing year
year = 2022
predicted_price = model.predict([[year]])[0]

print(f"Predicted price for a {year} car: {predicted_price:,.2f} million VND")
