import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load pre-trained model
model = joblib.load('diabetes_model.pkl')

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Handle zero Insulin to avoid division by zero
    insulin_val = Insulin if Insulin > 0 else 1
    glucose_insulin_ratio = Glucose / insulin_val
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                                BMI, DiabetesPedigreeFunction, Age, glucose_insulin_ratio]], 
                              columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                       'Insulin','BMI','DiabetesPedigreeFunction','Age','glucose_insulin_ratio'])
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    result = "Diabetic" if pred == 1 else "Non-Diabetic"
    confidence = f"{proba[pred]*100:.2f}%"
    return f"{result} (Confidence: {confidence})"

iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies", value=1),
        gr.Number(label="Glucose", value=100),
        gr.Number(label="BloodPressure", value=70),
        gr.Number(label="SkinThickness", value=20),
        gr.Number(label="Insulin", value=80),
        gr.Number(label="BMI", value=25),
        gr.Number(label="DiabetesPedigreeFunction", value=0.3),
        gr.Number(label="Age", value=30)
    ],
    outputs="text",
    title="Diabetes Prediction",
    description="Enter patient details to predict diabetes risk"
)

if __name__ == "__main__":
    iface.launch()