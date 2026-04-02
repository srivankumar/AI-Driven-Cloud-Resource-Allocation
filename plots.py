# plot.py

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("prediction_results.csv")

# Plot CPU
plt.figure()
plt.plot(data["Actual_CPU"], label="Actual CPU")
plt.plot(data["Predicted_CPU"], label="Predicted CPU")
plt.title("CPU Utilization Prediction")
plt.xlabel("Samples")
plt.ylabel("CPU Usage")
plt.legend()
plt.show() 

# Plot Memory
plt.figure()
plt.plot(data["Actual_Memory"], label="Actual Memory")
plt.plot(data["Predicted_Memory"], label="Predicted Memory")
plt.title("Memory Usage Prediction")
plt.xlabel("Samples")
plt.ylabel("Memory Usage")
plt.legend()
plt.show()
