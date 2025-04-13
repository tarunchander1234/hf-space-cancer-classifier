from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

app = FastAPI()

# Load model and tokenizer
model_path = "D:\Python_venv\Velsera\Code\saved_model\merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["Cancer", "Non-Cancer"]

class InferenceRequest(BaseModel):
    title: str
    abstract: str

@app.post("/predict")
def predict(request: InferenceRequest):
    text = f"{request.title} {request.abstract}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    
    predicted_idx = int(torch.argmax(torch.tensor(probs)))
    predicted_label = labels[predicted_idx]

    confidence_scores = {labels[i]: float(round(prob, 4)) for i, prob in enumerate(probs)}

    return {
        "predicted_labels": [predicted_label],
        "confidence_scores": confidence_scores
    }

# Optional if running locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)