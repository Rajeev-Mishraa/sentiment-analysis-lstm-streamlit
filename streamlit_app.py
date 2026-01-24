import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000//predict"

st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.write("Enter a review and get instant sentiment prediction using an LSTM model.")

# Input text
review = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text")
    else:
        response = requests.post(
            API_URL,
            json={"review": review}
        )

        if response.status_code == 200:
            result = response.json()
            sentiment = result["sentiment"]

            if sentiment == "Positive":
                st.success(f"✅ Sentiment: {sentiment}")
            else:
                st.error(f"❌ Sentiment: {sentiment}")
        else:
            st.error("API Error. Make sure FastAPI server is running.")

import streamlit as st
st.write("🚀 Streamlit is running")
