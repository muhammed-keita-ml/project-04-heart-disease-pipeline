import gradio as gr
import numpy as np
import joblib
import json

# Load model and feature names
model = joblib.load("heart_disease_model.pkl")
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


def predict_heart_disease(
    age, trestbps, chol, thalch, oldpeak, ca, sex, cp, fbs, restecg, exang, slope, thal
):

    # Start with all features set to 0
    input_dict = {col: 0 for col in feature_names}

    # Continuous features
    input_dict["age"] = age
    input_dict["trestbps"] = trestbps
    input_dict["chol"] = chol
    input_dict["thalch"] = thalch
    input_dict["oldpeak"] = oldpeak
    input_dict["ca"] = ca

    # Categorical features
    if sex == "Male":
        input_dict["sex_Male"] = 1

    cp_map = {
        "Atypical angina": "cp_atypical angina",
        "Non-anginal": "cp_non-anginal",
        "Typical angina": "cp_typical angina",
    }
    if cp in cp_map and cp_map[cp] in input_dict:
        input_dict[cp_map[cp]] = 1

    if fbs == "True (>120 mg/dl)":
        input_dict["fbs_True"] = 1

    restecg_map = {
        "Normal": "restecg_normal",
        "ST-T abnormality": "restecg_st-t abnormality",
    }
    if restecg in restecg_map and restecg_map[restecg] in input_dict:
        input_dict[restecg_map[restecg]] = 1

    if exang == "Yes":
        input_dict["exang_True"] = 1

    slope_map = {
        "Flat": "slope_flat",
        "Upsloping": "slope_upsloping",
    }
    if slope in slope_map and slope_map[slope] in input_dict:
        input_dict[slope_map[slope]] = 1

    thal_map = {
        "Normal": "thal_normal",
        "Reversable defect": "thal_reversable defect",
    }
    if thal in thal_map and thal_map[thal] in input_dict:
        input_dict[thal_map[thal]] = 1

    # Build feature array in correct column order
    input_array = np.array([[input_dict[col] for col in feature_names]])

    # Predict
    proba = model.predict_proba(input_array)[0][1]
    label = (
        "🔴 Heart Disease Likely" if proba >= 0.5 else "🟢 No Heart Disease Detected"
    )

    return f"{label}\n\nDisease probability: {proba:.1%}"


# Build Gradio interface
demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Slider(20, 80, value=50, step=1, label="Age"),
        gr.Slider(80, 200, value=130, step=1, label="Resting Blood Pressure (mmHg)"),
        gr.Slider(100, 600, value=240, step=1, label="Cholesterol (mg/dl)"),
        gr.Slider(60, 210, value=150, step=1, label="Max Heart Rate Achieved (bpm)"),
        gr.Slider(0, 7, value=1.0, step=0.1, label="ST Depression (oldpeak)"),
        gr.Slider(0, 3, value=0, step=1, label="Major Vessels Coloured (ca)"),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Dropdown(
            ["Asymptomatic", "Atypical angina", "Non-anginal", "Typical angina"],
            value="Asymptomatic",
            label="Chest Pain Type",
        ),
        gr.Radio(["True (>120 mg/dl)", "False"], label="Fasting Blood Sugar > 120?"),
        gr.Dropdown(
            ["Normal", "ST-T abnormality", "LV hypertrophy"],
            value="Normal",
            label="Resting ECG",
        ),
        gr.Radio(["Yes", "No"], label="Exercise-Induced Angina?"),
        gr.Dropdown(
            ["Flat", "Upsloping", "Downsloping"], value="Flat", label="ST Slope"
        ),
        gr.Dropdown(
            ["Normal", "Fixed defect", "Reversable defect"],
            value="Normal",
            label="Thalassemia",
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Risk Predictor",
    description=(
        "Enter patient clinical measurements to predict heart disease risk.\n"
        "Model: Random Forest | ROC-AUC: 0.921 | Disease Recall: 0.92\n"
        "Dataset: UCI Heart Disease (920 patients, 4 hospitals)\n"
        "⚠️ For educational purposes only — not a substitute for medical diagnosis."
    ),
    examples=[
        [
            63,
            145,
            233,
            150,
            2.3,
            0,
            "Male",
            "Typical angina",
            "False",
            "LV hypertrophy",
            "No",
            "Downsloping",
            "Fixed defect",
        ],
        [
            37,
            130,
            250,
            187,
            3.5,
            0,
            "Male",
            "Non-anginal",
            "False",
            "Normal",
            "No",
            "Upsloping",
            "Normal",
        ],
        [
            56,
            120,
            236,
            178,
            0.8,
            0,
            "Female",
            "Atypical angina",
            "False",
            "Normal",
            "No",
            "Upsloping",
            "Normal",
        ],
    ],
    theme=gr.themes.Soft(),
)

demo.launch()
