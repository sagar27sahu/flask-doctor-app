from flask import Flask, request, render_template, send_file, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Load the trained model
model = joblib.load("stacking_classifier.pkl")

# Load the dataset (Ensure this file exists in the same directory)
dataset_df = pd.read_excel("data.xlsx")

# Ensure "Login Hour" exists in dataset
dataset_df["Login Time"] = pd.to_datetime(dataset_df["Login Time"])  # Convert to datetime
dataset_df["Login Hour"] = dataset_df["Login Time"].dt.hour  # Extract hour

# Load Label Encoders
try:
    speciality_encoder = joblib.load("speciality_encoder.pkl")
    region_encoder = joblib.load("region_encoder.pkl")
    print("✅ Encoders loaded successfully!")
except FileNotFoundError:
    print("❌ Error: speciality_encoder.pkl or region_encoder.pkl not found! Train and save encoders first.")
    exit()

# Initialize Flask app
app = Flask(__name__)

# Function to process user input and predict doctors
def predict_doctors(time_input):
    # Convert input time to hour and weekday
    input_datetime = datetime.strptime(time_input, "%H:%M")
    input_hour = input_datetime.hour
    input_day = input_datetime.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if input_day >= 5 else 0  # Weekend flag

    # Extract relevant doctors based on input time
    doctors_available = dataset_df[dataset_df["Login Hour"] == input_hour]

    if doctors_available.empty:
        return pd.DataFrame(columns=["NPI", "Speciality", "Region", "Engagement Score"])

    # Encode categorical features
    doctors_available["Speciality"] = doctors_available["Speciality"].apply(
        lambda x: speciality_encoder.transform([x])[0] if x in speciality_encoder.classes_ else -1
    )
    doctors_available["Region"] = doctors_available["Region"].apply(
        lambda x: region_encoder.transform([x])[0] if x in region_encoder.classes_ else -1
    )

    # Ensure all necessary columns exist
    feature_columns = ["Login Hour", "Login Day", "Usage Time (mins)", "Speciality", 
                       "Region", "Survey Engagement Ratio", "Weekend", "Active Hour Bin", "Usage Category"]
    
    for col in feature_columns:
        if col not in doctors_available.columns:
            doctors_available[col] = 0  # Default value

    # Prepare data for prediction
    X_input = doctors_available[feature_columns]

    # Predict engagement likelihood
    predictions = model.predict_proba(X_input)[:, 1]  
    doctors_available["Engagement Score"] = predictions

    # Sort doctors by engagement probability
    doctors_sorted = doctors_available.sort_values(by="Engagement Score", ascending=False)

    return doctors_sorted[["NPI", "Speciality", "Region", "Engagement Score"]]

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for processing input and downloading CSV
@app.route("/predict", methods=["POST"])
def predict():
    time_input = request.form["time"]
    predicted_doctors = predict_doctors(time_input)
    
    if predicted_doctors.empty:
        return "No doctors found for the selected time.", 400
    
    # Save to CSV
    csv_filename = "predicted_doctors.csv"
    predicted_doctors.to_csv(csv_filename, index=False)
    
    return send_file(csv_filename, as_attachment=True)

# Route for Doctor Activity Trends (Graph Data)
@app.route("/doctor_trends")
def doctor_trends():
    # Count doctors active per hour
    hour_counts = dataset_df["Login Hour"].value_counts().sort_index()

    # Convert to JSON for Chart.js
    return jsonify({
        "hours": hour_counts.index.tolist(),
        "counts": hour_counts.values.tolist()
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)  # Change port if needed
