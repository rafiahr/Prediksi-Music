import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/linreg_model.pkl")

st.title("🎵 Prediksi Popularitas Lagu")
st.write("Masukkan fitur audio lagu berikut untuk memprediksi nilai popularitas (0-100):")

# Input fitur
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
duration_ms = st.number_input("Duration (ms)", min_value=0, value=180000)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=300.0, value=120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

# Prediksi
if st.button("Prediksi Popularitas"):
    input_features = np.array([[acousticness, danceability, duration_ms, energy,
                                instrumentalness, liveness, loudness, speechiness,
                                tempo, valence]])
    prediction = model.predict(input_features)[0]
    st.success(f"🎧 Perkiraan Popularitas Lagu: {prediction:.2f} / 100")