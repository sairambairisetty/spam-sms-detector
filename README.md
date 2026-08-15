# spam-sms-detector

# 📩 SMS Spam Detection System (End-to-End AI Web App)

An end-to-end Machine Learning web application designed to classify SMS messages into **SPAM** or **HAM** (Legitimate) in real time. It features a custom **PyTorch Neural Network** with a **TF-IDF Vectorizer** on the backend and a high-performance **Vite / Vanilla JavaScript** frontend.

---

## 💡 Why This Project? (Use Case)
SMS phishing, lottery scams, and spam messages are growing threats to digital security. This tool provides:
* **Instant Threat Detection:** Analyzes text patterns to filter out fraudulent promotions and phishing links.
* **Confidence Scoring:** Outputs exact probability scores alongside class predictions.
* **Ultra-Fast Inference:** Lightweight PyTorch inference engine served via FastAPI with minimal latency (<100ms).

---

## 🛠️ Architecture & Tech Stack
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/ce014ae4-55eb-404d-b7ea-044749cbb988" />


* **Frontend:** Vite, HTML5, CSS3, Modern JavaScript (ES6 Modules)
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** PyTorch (Feedforward Neural Network), Scikit-Learn (TF-IDF), Joblib/Pickle

---

## 📂 Project Directory Structure

```text
spam-sms-detector/
├── backend/
│   ├── models/
│   │   ├── spam_model.pth    # Trained PyTorch model weights
│   │   └── tfidf.pkl         # Fitted TF-IDF vectorizer
│   ├── model.py              # Neural network architecture definition
│   ├── train.py              # Model training script
│   ├── predict.py            # CLI prediction script
│   ├── main.py               # FastAPI application server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── main.js           # Frontend logic & API call handling
│   │   └── style.css         # UI styling
│   ├── index.html            # Entry web page
│   └── package.json          # Node dependencies and scripts
├── .gitignore
└── README.md

---

## 🚀 Step-by-Step Setup & Installation Guide

Follow these commands step-by-step to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/sairambairisetty/spam-sms-detector.git](https://github.com/sairambairisetty/spam-sms-detector.git)
cd spam-sms-detector


# Navigate to the backend directory
cd backend

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000



# Navigate to the frontend directory
cd spam-sms-detector/frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev



# Ensure backend virtual environment is active(optional)
cd backend
python train.py





