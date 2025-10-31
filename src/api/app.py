from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from models.TreeModel import load_pipeline, train_and_save_pipeline, predict_from_pipeline

app = FastAPI()


class Sample(BaseModel):
	Coffee_Intake: float
	Sleep_Hours: float
	Stress_Level: float
	Physical_Activity_Hours: float
	Age: float
	BMI: float
	Smoking: str
	Gender: str


try:
	pipeline = load_pipeline()
except Exception:
	train_and_save_pipeline(verbose=False)
	pipeline = load_pipeline()


@app.post('/predict')
def predict(sample: Sample):
	df = pd.DataFrame([sample.dict()])
	_, _, labels = predict_from_pipeline(df, pipeline)
	return {'label': labels[0]}
