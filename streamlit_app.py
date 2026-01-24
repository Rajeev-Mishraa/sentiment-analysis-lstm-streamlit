import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- Load Model & Tokenizer (cached) ----------
@st.cache_resource
def load_artifacts():
    model = load_model("sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ---------- UI ----------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review below and the model will predict whether the sentiment is **Positive** or **Negative**."
)

review = st.text_area(
    "✍️ Write your movie review here:",
    height=150,
    placeholder="The movie was amazing, with great performances and a strong storyline..."
)

# ---------- Prediction ----------
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before clicking predict.")
    else:
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=200)

        prediction = model.predict(padded)[0][0]

        st.markdown("---")
        if prediction > 0.5:
            st.success(f"✅ **Positive Sentiment** \n\nConfidence: `{prediction:.2f}`")
        else:
            st.error(f"❌ **Negative Sentiment** \n\nConfidence: `{1 - prediction:.2f}`")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ using LSTM, TensorFlow & Streamlit")
