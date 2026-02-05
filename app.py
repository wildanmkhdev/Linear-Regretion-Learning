import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🏠 Prediksi Harga Rumah kamu (Linear Regression)")

st.write("Masukkan luas rumah kamu, lalu klik Prediksi.")

luas = st.number_input("Luas rumah (m2):", min_value=0, value=50)

if st.button("Prediksi"):
    pred = model.predict([[luas]])
    hasil = pred.item()  
    st.success(f"Perkiraan harga rumah: {hasil:.2f}")
