import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("wireless_capacity_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Wireless Predictor", layout="centered")

st.markdown(
    "<h1 style='text-align:center;color:blue;'>📡 Wireless Capacity Predictor</h1>",
    unsafe_allow_html=True
)

st.write("Predict wireless channel capacity using trained ML model.")

# User Inputs
snr = st.slider("Enter Signal-to-Noise Ratio (SNR)", 0.0, 30.0, 10.0)

bandwidth = st.slider("Enter Bandwidth (MHz)", 1.0, 20.0, 5.0)

# Feature Engineering (same as notebook)
log_snr = np.log1p(snr)

interaction = snr * bandwidth

snr_squared = snr ** 2

# Prediction
input_data = pd.DataFrame([{
    'SNR': snr,
    'Bandwidth': bandwidth,
    'log_SNR': log_snr,
    'interaction': interaction,
    'snr_squared': snr_squared
}])

prediction = model.predict(input_data)

st.success(f"Predicted Channel Capacity: {prediction[0]:.2f}")