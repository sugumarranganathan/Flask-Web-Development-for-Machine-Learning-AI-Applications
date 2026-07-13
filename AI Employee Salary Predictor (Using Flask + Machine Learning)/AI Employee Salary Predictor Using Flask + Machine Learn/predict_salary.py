import pandas as pd
import joblib

print("Loading Saved AI Model...")

# Load Saved Model
model = joblib.load("salary_model.pkl")

print("Model Loaded Successfully!")

# New Employee
new_employee = pd.DataFrame({
    "Age": [35],
    "Experience": [10]
})

# Predict Salary
predicted_salary = model.predict(new_employee)

print("\nEmployee Details")
print("-----------------------")
print("Age :", new_employee["Age"][0])
print("Experience :", new_employee["Experience"][0], "Years")

print("\nPredicted Salary")
print("-----------------------")
print("₹{:,.2f}".format(predicted_salary[0]))