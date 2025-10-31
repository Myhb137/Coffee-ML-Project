"""Train and save a small pipeline (model + encoders) for sleep-quality prediction."""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


DATA_PATH = os.path.join('data', 'data.csv')
PIPELINE_PATH = os.path.join('model', 'tree_pipeline.pkl')
FEATURE_COLUMNS = [
    'Coffee_Intake', 'Sleep_Hours', 'Stress_Level',
    'Physical_Activity_Hours', 'Age', 'BMI', 'Smoking', 'Gender'
]


def train_and_save_pipeline(data_path=DATA_PATH, pipeline_path=PIPELINE_PATH, verbose=True):
    """Train model, store encoders and medians, and save pipeline dict."""
    df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Sleep_Quality'])

    encoders = {}
    medians = {}
    for col in FEATURE_COLUMNS:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            enc = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = enc.fit_transform(df[col])
            encoders[col] = enc
        else:
            medians[col] = df[col].median()

    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if verbose:
        print('\nModel performance summary:')
        print('-' * 40)
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"R2 : {r2_score(y_test, y_pred):.4f}")
        print(f"CV R2 (5-fold): {cross_val_score(model, X, y, cv=5, scoring='r2').mean():.4f}")
        print('-' * 40)

    pipeline = {
        'model': model,
        'label_encoder': label_encoder,
        'encoders': encoders,
        'feature_columns': FEATURE_COLUMNS,
        'medians': medians,
    }

    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    joblib.dump(pipeline, pipeline_path, compress=3)
    if verbose:
        print(f"Saved pipeline to: {pipeline_path}")


def load_pipeline(pipeline_path=PIPELINE_PATH):
    """Load the saved pipeline dict from disk."""
    return joblib.load(pipeline_path)


def preprocess_input(df_raw, pipeline):
    """Convert raw input to numeric matrix using saved encoders/medians."""
    df = df_raw.copy()
    feature_cols = pipeline['feature_columns']
    encs = pipeline['encoders']
    medians = pipeline['medians']

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    processed = pd.DataFrame()
    for col in feature_cols:
        if col in encs:
            raw_vals = df[col].astype(str).tolist()
            class_strs = [str(c) for c in encs[col].classes_]
            fallback = class_strs[0] if len(class_strs) > 0 else ''
            safe_vals = [v if v in class_strs else fallback for v in raw_vals]
            processed[col] = encs[col].transform(safe_vals)
        else:
            processed[col] = pd.to_numeric(df[col], errors='coerce')
            if processed[col].isna().any():
                median = medians.get(col, 0)
                processed[col].fillna(median, inplace=True)

    return processed.astype(float)


def predict_from_pipeline(df_raw, pipeline):
    """Return predicted numeric scores and decoded label values for input DataFrame."""
    proc = preprocess_input(df_raw, pipeline)
    scores = pipeline['model'].predict(proc)
    scores_rounded = np.round(scores).astype(int)
    labels = pipeline['label_encoder'].inverse_transform(scores_rounded)
    return scores, scores_rounded, labels


if __name__ == '__main__':
    # Train and save pipeline
    # Train pipeline without printing training metrics
    train_and_save_pipeline(verbose=False)

    # Load pipeline and run a single prediction
    pipeline = load_pipeline()
    sample = pd.DataFrame({
        'Coffee_Intake': [6],
        'Sleep_Hours': [6],
        'Stress_Level': [9],
        'Physical_Activity_Hours': [1.0],
        'Age': [22],
        'BMI': [22],
        'Smoking': ['Yes'],
        'Gender': ['Male'],
    })

    _, _, labels = predict_from_pipeline(sample, pipeline)
    # Print only the decoded label (single value)
    print(labels[0])


