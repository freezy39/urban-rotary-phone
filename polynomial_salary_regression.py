import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the salary dataset
# This file has two columns: YearsExperience and Salary
data = pd.read_csv(r"C:\Users\jash.farrell\Downloads\Salary_Data.csv")

# Separate the input (years of experience) from what we are trying to predict (salary)
X = data[['YearsExperience']]
y = data['Salary']

# Turn years of experience into polynomial features
# This lets the model learn a curve instead of forcing a straight line
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create and train the regression model
# At this point the model is learning how experience relates to pay
model = LinearRegression()
model.fit(X_poly, y)

# Print out what the model learned
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
print()

# Try out some example experience values to see if the predictions make sense
print("Sample Predictions:")
test_experience = [1, 3, 5, 8, 10]

for years in test_experience:
    years_array = np.array([[years]])
    years_poly = poly.transform(years_array)
    predicted_salary = model.predict(years_poly)[0]
    print(f"{years} years of experience -> Predicted salary: ${predicted_salary:,.2f}")

print("\nDone.")
