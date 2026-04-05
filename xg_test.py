

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert numeric columns
numeric_cols = [
    "cpu_utilization",
    "memory_usage",
    "storage_usage",
    "workload",
    "Resource_Allocation"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort
df = df.sort_values("timestamp").reset_index(drop=True)

# Handle missing values
df = df.ffill()

# -----------------------------
# Time features
# -----------------------------
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

# -----------------------------
# Lag features
# -----------------------------
for lag in [1,2,3,6,12]:
    df[f"cpu_lag_{lag}"] = df["cpu_utilization"].shift(lag)
    df[f"memory_lag_{lag}"] = df["memory_usage"].shift(lag)

df = df.dropna()

# -----------------------------
# Features & Target
# -----------------------------
features = [
    "workload","storage_usage","hour","day_of_week",
    "week_of_year","day_of_month","month","is_weekend"
]

for lag in [1,2,3,6,12]:
    features.append(f"cpu_lag_{lag}")
    features.append(f"memory_lag_{lag}")

X = df[features]
y = df[["cpu_utilization","memory_usage"]]

# -----------------------------
# 80-20 Split
# -----------------------------
split = int(len(df)*0.8)
X_test = X[split:]
y_test = y[split:]

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("final_xgboost_model.pkl")

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
cpu_mae = mean_absolute_error(y_test["cpu_utilization"], predictions[:,0])
memory_mae = mean_absolute_error(y_test["memory_usage"], predictions[:,1])

cpu_mse = mean_squared_error(y_test["cpu_utilization"], predictions[:,0])
memory_mse = mean_squared_error(y_test["memory_usage"], predictions[:,1])

cpu_rmse = np.sqrt(cpu_mse)
memory_rmse = np.sqrt(memory_mse)

cpu_r2 = r2_score(y_test["cpu_utilization"], predictions[:,0])
memory_r2 = r2_score(y_test["memory_usage"], predictions[:,1])

print("===== Model Performance =====")
print("CPU MAE:", cpu_mae)
print("Memory MAE:", memory_mae)
print("CPU MSE:", cpu_mse)
print("Memory MSE:", memory_mse)
print("CPU RMSE:", cpu_rmse)
print("Memory RMSE:", memory_rmse)
print("CPU R2 Score:", cpu_r2)
print("Memory R2 Score:", memory_r2)

# -----------------------------
# Comparison Output
# -----------------------------
comparison = pd.DataFrame({
    "Actual_CPU": y_test["cpu_utilization"].values,
    "Predicted_CPU": predictions[:,0],
    "Actual_Memory": y_test["memory_usage"].values,
    "Predicted_Memory": predictions[:,1]
})

print(" 10 Predictions:")
print(comparison.tail(10))

comparison.to_csv("prediction_results.csv", index=False)