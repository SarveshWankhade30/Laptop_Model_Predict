import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
model = joblib.load("laptop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# App title
st.title("Laptop Model Predictor")

# User input fields
brand = st.selectbox("Select Brand", ['Dell', 'HP', 'Lenovo', 'Apple', 'Acer', 'Asus', 'MSI', 'Razer', 'Microsoft', 'Samsung'])
processor = st.selectbox("Select Processor", ['Intel i3', 'Intel i5', 'Intel i7', 'Intel i9', 'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'M1', 'M2'])
ram = st.slider("RAM (GB)", 4, 32, 8, step=4)
storage = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048])

# Predict button
if st.button("Predict Laptop Model"):
    input_df = pd.DataFrame([[brand, processor, ram, storage]], columns=["Brand", "Processor", "RAM", "Storage"])
    pred_encoded = model.predict(input_df)
    pred_label = label_encoder.inverse_transform(pred_encoded)
    st.success(f"Predicted Laptop Model: {pred_label[0]}")
