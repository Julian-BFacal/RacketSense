import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load CSV data
csv_filename = '../data/SensorMango/13-03-25_15;51_azul/13-03-25_15;51_azul.csv'
df = pd.read_csv(csv_filename)
df = df[(df['timestamp'] >= 65.0) & (df['timestamp'] <= 165.0)]
df.reset_index(drop=True, inplace=True)

# Ensure timestamp is numeric
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Find peaks in accelx
df["accelx"] = df["accelx"].abs()
threshold = df['accelx'].max() * 0.4
print(threshold)
peaks, _ = find_peaks(df['accelx'], height=threshold, distance=60)

# Convert peak indices to timestamps
shots = []
prev_peak_time = None

for i, peak in enumerate(peaks):
    peak_time = float(df['timestamp'][peak])
    diff_with_prev = round(peak_time - prev_peak_time, 3) if prev_peak_time is not None else None
    prev_peak_time = peak_time

    shots.append({
        "shot_number": i + 1,
        "peak_time": round(peak_time, 3),
        "diff_with_prev": diff_with_prev
    })

# Save detected shots to JSON
json_filename = "../data/detected_shots_simple.json"
with open(json_filename, "w") as json_file:
    json.dump(shots, json_file, indent=4)

print(f"Detected shots saved to {json_filename}")

# Plot results
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['accelx'], label="AccelX")
plt.scatter(df['timestamp'][peaks], df['accelx'][peaks], color='r', label="Detected Peaks")
plt.axhline(threshold, color='g', linestyle='dashed', label="Threshold")
plt.xlabel("Timestamp")
plt.ylabel("AccelX")
plt.title("Detected Peaks in AccelX")
plt.legend()
plt.show()