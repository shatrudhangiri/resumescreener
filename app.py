from fastapi import FastAPI
import pickle
import os
import pickle
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working"}

# Input schema
class ResumeFile(BaseModel):
    resume_text: str

# Load vectorizer
with open('Models/tfidf_vectorizer_categorization.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load model
with open('Models/rf_classifier_categorization.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_category(data: ResumeFile):
    resume_text = data.resume_text
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    prediction = rf_classifier.predict(resume_tfidf)
    
    return {"predicted_category": prediction[0]}
