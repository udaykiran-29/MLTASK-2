import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Preparing the dataset
data = {
    "Feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Target": [2.1, 4.2, 6.1, 8.4, 10.2, 12.3, 14.5, 16.8, 18.7, 20.9]
}
df = pd.DataFrame(data)

# Step 2: Spliting the dataset into training and testing sets
X = df[["Feature"]]  # Feature
y = df["Target"]     # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluating the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 5: Visualizing the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
