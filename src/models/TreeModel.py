# Tree model for predicting sleep quality from lifestyle data
# Using decision trees since they handle both numeric and text data well
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and prep the data
data = pd.read_csv("data/data.csv")

# Convert text categories to numbers for the model
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Sleep_Quality"])

# Save encoders for each text column to use later
encoders = {}
categorical_cols = data.select_dtypes(include="object").columns

for col in categorical_cols:
    if col != "Sleep_Quality":
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col].astype(str))

# Pick features that affect sleep
X = data[['Coffee_Intake', 'Sleep_Hours', 'Stress_Level',
          'Physical_Activity_Hours', 'Age', 'BMI', 
          'Smoking', 'Gender']]

# ==========================================================
# 2. Train model
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)

# ==========================================================
# 3. Evaluate Model
# ==========================================================
y_pred = model.predict(X_test)

# Show how well the model performs
print("\nModel Performance:")
print("-" * 40)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score:           {r2_score(y_test, y_pred):.4f}")
print(f"Cross-Val R² Score: {cross_val_score(model, X, y, cv=5, scoring='r2').mean():.4f}")
print("-" * 40)

# ==========================================================
# 4. Make Predictions on New Data
# ==========================================================
sample_data = pd.DataFrame({
    "Coffee_Intake": [10],
    "Sleep_Hours": [8],
    "Stress_Level": [5],
    "Physical_Activity_Hours": [8],
    "Age": [22],
    "BMI": [26],
    "Smoking": ["Yes"],      # works regardless of type in dataset
    "Gender": ["Male"]
})

print("\nInput sample used for prediction:")
print(sample_data)
# Ensure sample columns match training features order
sample_data = sample_data.reindex(columns=X.columns)

# Encode / coerce each feature to the same dtype used in training
for col in X.columns:
    if col in encoders:
        # Transform categorical values using the same encoder; handle unseen values
        raw_vals = sample_data[col].astype(str).tolist()
        class_strs = [str(c) for c in encoders[col].classes_]
        fallback = class_strs[0] if len(class_strs) > 0 else ''
        replaced = [v if v in class_strs else fallback for v in raw_vals]
        unseen = sorted(set([v for v in raw_vals if v not in class_strs]))
        if unseen:
            print(f"Warning: unseen categories for column '{col}': {unseen}. Replacing with '{fallback}' before transform.")
        sample_data[col] = encoders[col].transform(replaced)
    else:
        # Numeric column: coerce to numeric and fill missing with training median
        sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce')
        if sample_data[col].isna().any():
            fallback_num = X[col].median()
            print(f"Warning: non-numeric values in column '{col}' were coerced to NaN. Filling with median: {fallback_num}.")
            sample_data[col].fillna(fallback_num, inplace=True)

# Ensure all columns are numeric floats for the model
sample_data = sample_data.astype(float)

# Predict
pred = np.round(model.predict(sample_data)).astype(int)

# Get the prediction in readable form
predicted_quality = label_encoder.inverse_transform(pred)[0]

# Show input data and prediction
print("\nInput Values:")
print("-" * 40)
for col in X.columns:
    val = sample_data[col].values[0]
    print(f"{col:25}: {val}")
print("-" * 40)

print("\nResults:")
print("-" * 40)
print(f"Predicted Sleep Quality: {predicted_quality}")
print("-" * 40)


