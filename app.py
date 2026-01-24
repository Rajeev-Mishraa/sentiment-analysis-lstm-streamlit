from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Load model
model = load_model("sentiment_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

# Request body schema
class ReviewRequest(BaseModel):
    review: str

def predict_sentiment(review: str):
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred = model.predict(padded, verbose=0)
    return "Positive" if pred[0][0] > 0.5 else "Negative"

@app.post("/predict")
def predict(request: ReviewRequest):
    sentiment = predict_sentiment(request.review)
    return {
        "review": request.review,
        "sentiment": sentiment
    }