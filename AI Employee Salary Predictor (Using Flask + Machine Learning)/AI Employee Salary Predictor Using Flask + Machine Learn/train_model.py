import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Read Dataset
data = pd.read_csv("employee_salary.csv")

# Features (Input)
X = data[["Age", "Experience"]]

# Target (Output)
y = data["Salary"]

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

print("===================================")
print("AI Model Trained Successfully!")
print("===================================")

# Save Model
joblib.dump(model, "salary_model.pkl")

print("\nModel Saved Successfully!")
print("File Name : salary_model.pkl")

# New Employee
new_employee = pd.DataFrame({
    "Age": [30],
    "Experience": [6]
})

# Predict Salary
predicted_salary = model.predict(new_employee)

print("\nNew Employee Details")
print("-----------------------")
print("Age :", new_employee["Age"][0])
print("Experience :", new_employee["Experience"][0], "Years")

print("\nPredicted Salary")
print("-----------------------")
print("₹{:,.2f}".format(predicted_salary[0]))