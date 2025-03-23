import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="AI Medical Diagnosis", page_icon="🩺", layout="wide")

# Custom CSS Styling
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background-color: black;
    padding: 15px;
    border-radius: 15px;
}}

.stButton>button {{
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    transition: 0.3s;
    box-shadow: 2px 2px 10px rgba(0, 255, 0, 0.5);
}}

.stButton>button:hover {{
    background-color: #45a049;
    box-shadow: 2px 2px 12px rgba(0, 255, 0, 0.8);
}}

input, select, textarea, .stNumberInput, .stRadio, .stSlider {{
    background-color: black !important;
    color: white !important;
    padding: 12px;
    border-radius: 8px;
    box-shadow: 2px 2px 5px rgba(255, 255, 255, 0.2);
    border: 1px solid #ccc;
}}

h1, h2, h3 {{
    color: white;
    text-align: center;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0.8);
    padding: 15px;
    border-radius: 12px;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🩺 AI-Powered Medical Diagnosis System")

# Load models
models = {
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb')),
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb'))
}

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Heart Disease", "Parkinson's", "Lung Cancer", "Hypothyroid", "Diabetes"],
    icons=["heart", "brain", "lungs", "thermometer", "droplet"],
    orientation="horizontal",
    styles={"container": {"padding": "10px", "border-radius": "10px"}}
)

def predict(model, inputs):
    prediction = model.predict(np.array(inputs).reshape(1, -1))
    return "✅ No Disease Detected" if prediction[0] == 0 else "⚠️ Disease Detected"

# UI Design
if selected == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    features = [st.number_input(feature) for feature in [
        "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol",
        "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate Achieved", "Exercise-Induced Angina",
        "ST Depression", "Slope of ST Segment", "Major Vessels Colored by Fluoroscopy", "Thalassemia Type"
    ]]
    
    if st.button("Predict Heart Disease"):
        result = predict(models['heart_disease'], features)
        st.success(result)

elif selected == "Parkinson's":
    st.header("🧠 Parkinson's Disease Prediction")
    features = [st.number_input(feature) for feature in [
        "MDVP:Fo (Hz)", "MDVP:Fhi (Hz)", "MDVP:Flo (Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "Spread1",
        "Spread2", "D2", "PPE"
    ]]
    
    if st.button("Predict Parkinson's"):
        result = predict(models['parkinsons'], features)
        st.success(result)

elif selected == "Lung Cancer":
    st.header("🫁 Lung Cancer Prediction")
    features = [st.number_input(feature) for feature in [
        "Age", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure", "Chronic Disease",
        "Fatigue", "Allergy", "Wheezing", "Alcohol Consumption", "Coughing", "Shortness of Breath",
        "Swallowing Difficulty", "Chest Pain"
    ]]
    
    if st.button("Predict Lung Cancer"):
        result = predict(models['lung_cancer'], features)
        st.success(result)

elif selected == "Hypothyroid":
    st.header("🌡️ Hypothyroid Prediction")
    features = [st.number_input(feature) for feature in [
        "TSH", "T3", "TT4", "T4U", "FTI", "On Thyroxine", "Query Hypothyroid"
    ]]
    
    if st.button("Predict Hypothyroid"):
        result = predict(models['thyroid'], features)
        st.success(result)

elif selected == "Diabetes":
    st.header("🩸 Diabetes Prediction")
    features = [st.number_input(feature) for feature in [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin",
        "BMI", "Diabetes Pedigree Function", "Age"
    ]]
    
    if st.button("Predict Diabetes"):
        result = predict(models['diabetes'], features)
        st.success(result)
