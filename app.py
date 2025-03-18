from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ✅ Convert range strings (e.g., "4-6") to numeric values
def convert_to_numeric(value):
    try:
        if "-" in value:
            parts = value.split("-")
            return (float(parts[0]) + float(parts[1])) / 2  # Take average
        return float(value)  # Convert directly if single value
    except ValueError:
        return None  # Return None if conversion fails

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get form data
        gender = 1 if request.form["gender"] == "Male" else 0
        age = convert_to_numeric(request.form["age"])
        academic_year = convert_to_numeric(request.form["academic_year"])
        average_sleep = convert_to_numeric(request.form["average_sleep"])
        depression = convert_to_numeric(request.form["depression"])
        anxiety = convert_to_numeric(request.form["anxiety"])
        cgpa = convert_to_numeric(request.form["cgpa"])
        stress_level = convert_to_numeric(request.form["stress_level"])

        # ✅ Validate input (ensure no missing values)
        if None in [age, academic_year, average_sleep, depression, anxiety, cgpa, stress_level]:
            return render_template("index.html", prediction="Invalid input. Please enter valid numbers.")

        # ✅ Create input array with all 8 features
        features = np.array([[gender, age, academic_year, average_sleep, depression, anxiety, cgpa, stress_level]])

        # ✅ Scale input
        features_scaled = scaler.transform(features)

        # ✅ Make prediction
        prediction = model.predict(features_scaled)[0]

        return render_template("index.html", prediction=f"Predicted Stress Level: {int(prediction)}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)



