# train.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert numeric columns safely
numeric_cols = [
    "cpu_utilization",
    "memory_usage",
    "storage_usage",
    "workload",
    "Resource_Allocation"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# Handle missing values (important for time series)
df = df.ffill()

# -----------------------------
# Time Features
# -----------------------------
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

# -----------------------------
# Lag Features
# -----------------------------
for lag in [1, 2, 3, 6, 12]:
    df[f"cpu_lag_{lag}"] = df["cpu_utilization"].shift(lag)
    df[f"memory_lag_{lag}"] = df["memory_usage"].shift(lag)

# Drop rows created due to lag
df = df.dropna()

# -----------------------------
# Features & Target
# -----------------------------
features = [
    "workload",
    "storage_usage",
    "hour",
    "day_of_week",
    "week_of_year",
    "day_of_month",
    "month",
    "is_weekend"
]

for lag in [1, 2, 3, 6, 12]:
    features.append(f"cpu_lag_{lag}")
    features.append(f"memory_lag_{lag}")

X = df[features]
y = df[["cpu_utilization", "memory_usage"]]

# -----------------------------
# Train Model on FULL DATA
# -----------------------------
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
)

model.fit(X, y)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "final_xgboost_model.pkl")

print("Model trained and saved successfully.")