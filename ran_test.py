# plot_rf.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Handle missing values
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
model = joblib.load("rf_cpu_memory_model.pkl")
predictions = model.predict(X_test)

# -----------------------------
# 1. CPU Plot
# -----------------------------
plt.figure()
plt.plot(y_test["cpu_utilization"].values, label="Actual CPU")
plt.plot(predictions[:,0], label="Predicted CPU")
plt.title("CPU Utilization: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("CPU Usage")
plt.legend()
plt.show()

# -----------------------------
# 2. Memory Plot
# -----------------------------
plt.figure()
plt.plot(y_test["memory_usage"].values, label="Actual Memory")
plt.plot(predictions[:,1], label="Predicted Memory")
plt.title("Memory Usage: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Memory Usage")
plt.legend()
plt.show()

# -----------------------------
# 3. CPU Error Plot
# -----------------------------
cpu_error = y_test["cpu_utilization"].values - predictions[:,0]

plt.figure()
plt.plot(cpu_error)
plt.title("CPU Prediction Error")
plt.xlabel("Test Samples")
plt.ylabel("Error")
plt.show()

# -----------------------------
# 4. Memory Error Plot
# -----------------------------
memory_error = y_test["memory_usage"].values - predictions[:,1]

plt.figure()
plt.plot(memory_error)
plt.title("Memory Prediction Error")
plt.xlabel("Test Samples")
plt.ylabel("Error")
plt.show()