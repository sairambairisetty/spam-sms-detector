from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
from model import Spam

app = FastAPI()

# Enable CORS so your Vite frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TF-IDF Vectorizer
tfidf = joblib.load("models/tfidf.pkl")

# Load PyTorch Model
input_dim = len(tfidf.get_feature_names_out())
model = Spam(input_dim=input_dim)
model.load_state_dict(torch.load("models/spam_model.pth", map_location=torch.device("cpu")))
model.eval()

# Request body schema
class MessageRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(request: MessageRequest):
    text = request.text
    
    with torch.no_grad():
        vec = tfidf.transform([text]).toarray()
        tensor_input = torch.tensor(vec, dtype=torch.float32)
        prob = model(tensor_input).item()

    # Threshold set to 0.75
    threshold = 0.75
    if prob >= threshold:
        label = "SPAM"
        confidence = prob * 100
    else:
        label = "HAM"
        confidence = (1 - prob) * 100

    return {
        "result": f"{label} ({confidence:.1f}%)",
        "label": label,
        "confidence": round(confidence, 1)
    }