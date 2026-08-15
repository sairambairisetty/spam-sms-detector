import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from model import Spam


os.makedirs("models", exist_ok=True)


df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
messages = df["v2"].tolist()
labels = df["v1"].map({"ham": 0, "spam": 1}).tolist()


tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
X = tfidf.fit_transform(messages).toarray()

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)


model = Spam(input_dim=X_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training started...")
for epoch in range(500):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), "models/spam_model.pth")
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Training complete! Files saved in 'models/' folder.")