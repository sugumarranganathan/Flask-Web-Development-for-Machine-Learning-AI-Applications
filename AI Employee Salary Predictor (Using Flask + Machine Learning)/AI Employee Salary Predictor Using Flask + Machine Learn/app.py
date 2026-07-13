from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
# Load Trained AI Model
model = joblib.load("salary_model.pkl")

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Receive Form Data
@app.route("/predict", methods=["POST"])
def predict():

    employee_name = request.form["name"]
    age = int(request.form["age"])
    education = request.form["education"]
    experience = int(request.form["experience"])
    job_role = request.form["job_role"]

    print("Employee Name :", employee_name)
    print("Age :", age)
    print("Education :", education)
    print("Experience :", experience)
    print("Job Role :", job_role)
    
     
     # Prepare Data for AI Model
    employee = pd.DataFrame({
        "Age": [age],
        "Experience": [experience]
    })

    # Predict Salary
    predicted_salary = model.predict(employee)[0]

    print("Predicted Salary :", predicted_salary)

    return render_template(
    "result.html",
    employee_name=employee_name,
    age=age,
    education=education,
    experience=experience,
    job_role=job_role,
    predicted_salary=f"₹{predicted_salary:,.2f}"
)

if __name__ == "__main__":
    app.run(debug=True)