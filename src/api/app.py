from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import pandas as pd

# Fix imports for your project structure
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.TreeModel import load_pipeline, train_and_save_pipeline, predict_from_pipeline

# Initialize FastAPI app
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can replace "*" with your frontend URL (e.g., "http://localhost:5173")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class Sample(BaseModel):
    Coffee_Intake: float
    Sleep_Hours: float
    Stress_Level: float
    Physical_Activity_Hours: float
    Age: float
    BMI: float
    Smoking: str
    Gender: str

# Load or train the model
try:
    pipeline = load_pipeline()
except Exception:
    train_and_save_pipeline(verbose=False)
    pipeline = load_pipeline()

# Prediction endpoint
@app.post("/predict")
def predict(sample: Sample):
    df = pd.DataFrame([sample.dict()])
    _, _, labels = predict_from_pipeline(df, pipeline)
    return {"label": labels[0]}
