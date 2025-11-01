from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
import pandas as pd

# --- Fix imports for your project structure ---
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.TreeModel import load_pipeline, train_and_save_pipeline, predict_from_pipeline

# --- Initialize FastAPI app ---
app = FastAPI(title="Sleep Quality Predictor API")

# --- Allow frontend requests (CORS setup) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL in production (e.g., "http://localhost:5173")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Input schema for POST requests ---
class Sample(BaseModel):
    Coffee_Intake: float
    Sleep_Hours: float
    Stress_Level: float
    Physical_Activity_Hours: float
    Age: float
    BMI: float
    Smoking: str
    Gender: str

# --- Load or train your ML model ---
try:
    pipeline = load_pipeline()
except Exception:
    train_and_save_pipeline(verbose=False)
    pipeline = load_pipeline()

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "The Sleep Quality service is running."}

# --- POST prediction endpoint (JSON body) ---
@app.post("/predict")
def predict_post(sample: Sample):
    df = pd.DataFrame([sample.dict()])
    _, _, labels = predict_from_pipeline(df, pipeline)
    return {"label": labels[0]}

# --- GET prediction endpoint (query parameters) ---
@app.get("/predict")
def predict_get(
    Coffee_Intake: float,
    Sleep_Hours: float,
    Stress_Level: float,
    Physical_Activity_Hours: float,
    Age: float,
    BMI: float,
    Smoking: str,
    Gender: str
):
    df = pd.DataFrame([{
        "Coffee_Intake": Coffee_Intake,
        "Sleep_Hours": Sleep_Hours,
        "Stress_Level": Stress_Level,
        "Physical_Activity_Hours": Physical_Activity_Hours,
        "Age": Age,
        "BMI": BMI,
        "Smoking": Smoking,
        "Gender": Gender
    }])
    _, _, labels = predict_from_pipeline(df, pipeline)
    return {"label": labels[0]}
