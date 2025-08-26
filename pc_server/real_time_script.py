import pandas as pd
import numpy as np
import json
import sys
import os
import datetime
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
import joblib  # Para cargar modelo si ya está entrenado
import matplotlib.pyplot as plt
from glob import glob

# ================= CONFIGURACIÓN =================
csv_files = sorted(glob("./data/imu_*.csv"))
if not csv_files:
    raise FileNotFoundError("❌ No se encontró ningún archivo imu_*.csv en ./data/")

session_timestamp = sys.argv[1]
RAW_CSV = f"./data/imu_{session_timestamp}.csv"  # Último por orden alfabético (timestamp en nombre)
print(f"📥 Usando archivo IMU: {RAW_CSV} # Archivo CSV generado por el sistema IMU")
MODEL_PATH = "./models/rf_model.pkl"  # Modelo entrenado previamente
OUTPUT_DIR = Path("./sessions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
session_timestamp = sys.argv[1]
filename = f"imu_{session_timestamp}.csv"
session_name = filename.replace("imu_", "session_")
session_folder = OUTPUT_DIR / session_name
session_folder.mkdir(parents=True, exist_ok=True)


# ================ CARGA DE DATOS =================
df = pd.read_csv(RAW_CSV)
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ================ PARÁMETROS =====================
fs = 1 / df['timestamp'].diff().mean()  # Frecuencia de muestreo
min_distance_samples = int(fs)
local_window_size = int(0.5 * fs)
max_diff = 0.5

# ================ DETECCIÓN DE GOLPES ============
resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)

thresholdaccel = np.max(resultant_accel) * 0.35
thresholdgyro = np.max(gyro_magnitude) * 0.35

peaks, _ = find_peaks(resultant_accel, height=thresholdaccel, distance=min_distance_samples)
gyropeaks, _ = find_peaks(gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)

shots = []
prev_peak_time = None
gyropeak_times = df['timestamp'][gyropeaks].values
accelpeak_times = df['timestamp'][peaks].values

def get_local_max(series, index, window_size):
    start = max(0, index - window_size)
    end = min(len(series), index + window_size)
    return np.max(series[start:end]) if start < end else series[index]

def calculate_confidence(missing_series, peak_idx, mean_value, threshold, local_window_size):
    if peak_idx is None:
        return 0
    local_max = get_local_max(missing_series, peak_idx, local_window_size)
    if local_max <= mean_value:
        return 0
    if local_max >= threshold:
        return 100
    confidence = ((local_max - mean_value) / (threshold - mean_value)) * 100
    return max(0, min(abs(confidence), 100))

for i, peak in enumerate(peaks):
    peak_time = float(df['timestamp'][peak])
    nearby_gyro_idx = np.where(np.abs(gyropeak_times - peak_time) <= max_diff)[0]
    if len(nearby_gyro_idx) > 0:
        confidence = 100
        source = "both"
    else:
        confidence = calculate_confidence(gyro_magnitude, peak, np.mean(gyro_magnitude), thresholdgyro, local_window_size)
        source = "accel"
    if confidence > 70:
        shot_data = {
            "peak_time": round(peak_time, 3),
            "confidence": round(confidence, 3),
            "source": source
        }
        if prev_peak_time is not None:
            shot_data["diff_from_prev"] = round(peak_time - prev_peak_time, 3)
        else:
            shot_data["diff_from_prev"] = None
        prev_peak_time = peak_time
        shots.append(shot_data)

for i, gyro_peak in enumerate(gyropeaks):
    if not any(np.abs(accelpeak_times - df['timestamp'][gyro_peak]) <= max_diff):
        peak_time = float(df['timestamp'][gyro_peak])
        confidence = calculate_confidence(resultant_accel, gyro_peak, np.mean(resultant_accel), thresholdaccel, local_window_size)
        if confidence > 70:
            shot_data = {
                "peak_time": round(peak_time, 3),
                "confidence": round(confidence, 3),
                "source": "gyro"
            }
            if prev_peak_time is not None:
                shot_data["diff_from_prev"] = round(peak_time - prev_peak_time, 3)
            else:
                shot_data["diff_from_prev"] = None
            prev_peak_time = peak_time
            shots.append(shot_data)
shots = sorted(shots, key=lambda shot: shot["peak_time"])

# ============ SEGMENTACIÓN Y EXTRACCIÓN DE FEATURES ============
def segment_data_around_peaks(df, shots, fs):
    """
    Segment the data into strokes around the detected impact peaks using a 1-second window
    (0.5s before and 0.5s after the peak).
    """
    half_window = int(0.5 * fs)
    stroke_data = []
    for shot in shots:
        peak_time = shot["peak_time"]
        peak_idx = df[df['timestamp'] >= peak_time].index[0]
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(df), peak_idx + half_window)
        stroke_segment = df.iloc[start_idx:end_idx]
        stroke_data.append(stroke_segment)
    return stroke_data

def extract_features(stroke_segment):
    features = []
    for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
        features.extend([
            np.mean(stroke_segment[col]), np.std(stroke_segment[col]),
            skew(stroke_segment[col]), kurtosis(stroke_segment[col]),
            np.min(stroke_segment[col]), np.max(stroke_segment[col])
        ])
    return features

stroke_segments = segment_data_around_peaks(df, shots, fs)
features = [extract_features(seg) for seg in stroke_segments]

# ============ CLASIFICACIÓN =================
clf = joblib.load(MODEL_PATH)  # Modelo previamente entrenado
predictions = clf.predict(features)
label_map = {0: "serve", 1: "forehand", 2: "backhand"}
predicted_labels = [label_map.get(label, "unknown") for label in predictions]

# ============ GUARDAR RESULTADOS ============
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Guardar CSV raw
df.to_csv(session_folder / "raw.csv", index=False)

# Guardar shots detectados
with open(session_folder / "detected.json", "w") as f:
    json.dump(shots, f, indent=4)

# Guardar clasificación
classified = [{"peak_time": shot["peak_time"], "label": label} for shot, label in zip(shots, predicted_labels)]
with open(session_folder / "classified.json", "w") as f:
    json.dump(classified, f, indent=4)

fig, ax = plt.subplots(figsize=(12, 6))
accel_magnitude = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
ax.plot(df['timestamp'], accel_magnitude, label="Accel Magnitude", alpha=0.8)
for shot in shots:
    if shot['confidence'] > 70:
        peak_time = shot['peak_time']
        peak_idx = (np.abs(df['timestamp'] - peak_time)).argmin()
        peak_val = accel_magnitude[peak_idx]
        color = 'red' if shot['source'] == 'both' else 'red'
        ax.plot(peak_time, peak_val, 'o', color=color, markersize=10)  # círculo visible
        


handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

ax.set_title("Golpes detectados (confianza > 70)")
ax.set_xlabel("Timestamp (s)")
ax.set_ylabel("Acelerómetro (magnitud)")
ax.grid(True)

# Guardar imagen
plot_path = session_folder / "shots.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"🖼️ Imagen guardada en: {plot_path}")
print(f"✅ Resultados guardados en: {session_folder}")

from drive_uploader import upload_session_to_drive

upload_session_to_drive(str(session_folder))
print("📤 Carpeta subida a Drive")
