import pickle
import torch
from model import Spam

class FastSpamPredictor:
    def __init__(self, model_path="models/spam_model.pth", tfidf_path="models/tfidf.pkl"):
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)

        input_dim = len(self.tfidf.get_feature_names_out())
        self.model = Spam(input_dim=input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def predict(self, text: str) -> str:
        with torch.no_grad():
            vec = self.tfidf.transform([text]).toarray()
            tensor_input = torch.tensor(vec, dtype=torch.float32)
            prob = self.model(tensor_input).item()

        # Simple original condition without custom threshold
        if prob > 0.5:
            return f"SPAM ({prob * 100:.1f}%)"
        else:
            return f"HAM ({(1 - prob) * 100:.1f}%)"

if __name__ == "__main__":
    detector = FastSpamPredictor()
    while True:
        msg = input("\nEnter message to test (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break
        print("Prediction:", detector.predict(msg))