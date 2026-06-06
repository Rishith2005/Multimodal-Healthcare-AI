import requests
import json

url = "http://localhost:8000/api/predict/tabular"

payload_healthy = {
    "PatientName": "Jane Doe",
    "Gender": "Female",
    "Age": 24.0,
    "Height": 168.0,
    "Weight": 58.0,
    "BloodGlucose": 85.0,
    "BloodPressure": 70.0,
    "FamilyHistory": "No Family History",
    "PhysicalActivity": "Active (Moderate)",
    "SmokingStatus": "Never Smoked",
    "MedicalConditions": "None",
    "Symptoms": [],
    "HbA1c": 5.2,
    "InsulinLevel": 75.0,
    # Advanced optional
    "Pregnancies": 0.0,
    "SkinThickness": 18.0,
    "DiabetesPedigreeFunction": 0.24
}

payload_atrisk = {
    "PatientName": "John Smith",
    "Gender": "Male",
    "Age": 46.0,
    "Height": 175.0,
    "Weight": 105.0,
    "BloodGlucose": 165.0,
    "BloodPressure": 88.0,
    "FamilyHistory": "First-Degree Relative",
    "PhysicalActivity": "Sedentary (Low)",
    "SmokingStatus": "Former Smoker",
    "MedicalConditions": "Hypertension",
    "Symptoms": ["Polyuria", "Polydipsia", "Fatigue"],
    "HbA1c": 7.2,
    "InsulinLevel": 185.0,
    # Advanced optional
    "DiabetesPedigreeFunction": 0.72
}

print("Testing Healthy Patient...")
r = requests.post(url, json=payload_healthy)
print("Status:", r.status_code)
print("Response:", json.dumps(r.json(), indent=2))

print("\nTesting At Risk Patient...")
r = requests.post(url, json=payload_atrisk)
print("Status:", r.status_code)
print("Response:", json.dumps(r.json(), indent=2))
