from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Load the model
model_path = "../model/Hypermeter_tuning/best_model_lr2e-05_bs16_fold1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion labels
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

app = FastAPI(title="Emotion Prediction API")

# Frontend page
frontend_folder = "../front_end"  # folder where the HTML file is located
app.mount("/static", StaticFiles(directory=frontend_folder), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_folder, "main_page.html"))

class TextInput(BaseModel):
    texts: List[str]  # batch of texts

@app.post("/predict")
def predict(input_data: TextInput):
    texts = input_data.texts

    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()

    # Convert to predictions
    preds = (probs >= 0.5).astype(int)

    results = []
    for i, text in enumerate(texts):
        pred_labels = [emotion_columns[j] for j, v in enumerate(preds[i]) if v == 1]
        probs_dict = {emotion_columns[j]: float(probs[i][j]) for j in range(len(emotion_columns))}
        results.append({
            "text": text,
            "predicted_labels": pred_labels,
            "probabilities": probs_dict
        })

    return {"predictions": results}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
