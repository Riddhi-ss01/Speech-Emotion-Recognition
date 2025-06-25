# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
from utils import extract_deep_features
import os

# Load Model
@st.cache_resource
def load_emotion_model():
    model = load_model('model/emotion_model.keras')
    return model

# Emotion Classes
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Streamlit App
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎙️ Speech Emotion Recognition Web App")

st.write("Upload an audio file (WAV format) and get the predicted emotion!")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_deep_features("temp_audio.wav")
    if features is not None:
        model = load_emotion_model()
        prediction = model.predict(features)
        predicted_label = emotion_labels[np.argmax(prediction)]

        st.success(f"### 🎯 Predicted Emotion: **{predicted_label.upper()}**")

        st.write("### Prediction Probabilities:")
        for label, prob in zip(emotion_labels, prediction[0]):
            st.write(f"- {label.capitalize()}: {prob:.2%}")
    else:
        st.error("Error processing the audio file. Please try another file.")
