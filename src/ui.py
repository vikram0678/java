import streamlit as st
import requests
import os

st.title("🐦 BERT Sentiment Analysis MLOps Hub")

# Read API URL from environment variable or default to local Java API
api_url = os.getenv("API_URL", "http://localhost:8000/api/v1/sentiment/predict")

text = st.text_input("Enter text for sentiment analysis:", "This project configuration is amazing!")

if st.button("Analyze Sentiment"):
    if text.strip():
        try:
            res = requests.post(api_url, json={"text": text})
            if res.status_code == 200:
                result = res.json()
                st.success(f"Prediction: {result.get('label')}")
            else:
                st.error(f"Error from Java API server: {res.status_code}")
        except Exception as e:
            st.error(f"Failed to connect to backend service: {e}")
    else:
        st.warning("Please enter some text first.")
