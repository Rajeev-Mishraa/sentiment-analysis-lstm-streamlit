import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cache model loading
@st.cache_resource
def load_sentiment_model():
    return load_model("sentiment_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_sentiment_model()
tokenizer = load_tokenizer()

# UI
st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200, padding="post")
        prediction = model.predict(padded)[0][0]

        if prediction > 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")
