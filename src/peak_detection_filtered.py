import joblib
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from hmmlearn import hmm


# ---------------------- CONFIG ----------------------
JSON_FILE = "../data/SensorTriangulo/26-04-25_10;35/video_shots_26-04-25_10;35.json"
CSV_FILE = '../data/dentro1.csv'
xmin, xmax = 0, 150
#13-03-25_15;51_amarilla 76,178
#13-03-25_16;00_amarilla 25,130
#13-03-25_15;51_azul 65,165
#13-03-25_16;00_azul 32,127

# ---------------------- CONSTANTS ----------------------
FEATURE_LABELS = [
    f"{sensor}{stat}" for sensor in ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    for stat in ['Mean', 'SD', 'Skew', 'Kurtosis', 'Min', 'Max']
]


df = pd.read_csv(CSV_FILE)
df = df[(df['timestamp'] >= xmin) & (df['timestamp'] <= xmax)]
df_filtered = df.copy()
df.reset_index(drop=True, inplace=True)

# Ensure timestamp is in the correct format
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

def mean_timestamp_difference(df):
    df['timestamp_diff'] = df['timestamp'].diff()
    return df['timestamp_diff'].mean()

fs = 1 / mean_timestamp_difference(df)
min_distance_samples = fs
local_window_size = int(0.5 * fs)  # Adjust based on sampling rate
max_diff = 0.5 # Time threshold for considering peaks as paired

"""
# Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff=6, fs=60, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply Low-Pass Filter
for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
    df_filtered[col] = butter_lowpass_filter(df[col])

# Wavelet Denoising Function
def wavelet_denoise(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c), mode='soft') for c in coeffs[1:]]
    denoised_data = pywt.waverec(coeffs, wavelet)

    # Ensure the denoised data has the same length as the input data
    if len(denoised_data) != len(data):
        denoised_data = denoised_data[:len(data)]  # Trim the data if it's longer
        # Alternatively, use padding if the data is shorter:
        # denoised_data = np.pad(denoised_data, (0, len(data) - len(denoised_data)), mode='constant')
    return denoised_data

# Apply Wavelet Denoising
for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
    df_filtered[col] = wavelet_denoise(df[col])
"""
# Compute resultant acceleration
resultant_accel = np.sqrt(df_filtered['accelx'] ** 2 + df_filtered['accely'] ** 2 + df_filtered['accelz'] ** 2)
gyro_magnitude = np.sqrt(df_filtered['gyrox'] ** 2 + df_filtered['gyroy'] ** 2 + df_filtered['gyroz'] ** 2)

thresholdaccel = np.max(resultant_accel) * 0.35
thresholdgyro = np.max(gyro_magnitude) * 0.35

# Detect peaks
peaks, _ = find_peaks(resultant_accel, height=thresholdaccel, distance=min_distance_samples)
gyropeaks, _ = find_peaks(gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)

shots = []
prev_peak_time = None

# Convert peak indices to timestamps
gyropeak_times = df['timestamp'][gyropeaks].values
accelpeak_times = df['timestamp'][peaks].values

def get_local_max(series, index, window_size):
    """ Get local max in a small window around the given index. """
    if index is None:
        return 0  # No valid peak
    start = max(0, index - window_size)
    end = min(len(series), index + window_size)
    return np.max(series[start:end]) if start < end else series[index]

def calculate_confidence(missing_series, peak_idx, mean_value, threshold, local_window_size):
    """ Compute confidence scaled between mean value and threshold. """
    if peak_idx is None:
        return 0

    local_max = get_local_max(missing_series, peak_idx, local_window_size)

    # Scale confidence: 0% at mean value, 100% at threshold
    if local_max <= mean_value:
        return 0
    if local_max >= threshold:
        return 100

    confidence = ((local_max - mean_value) / (threshold - mean_value)) * 100
    return max(0, min(abs(confidence), 100))  # Ensure within 0-100 range

for i, peak in enumerate(peaks):
    peak_time = float(df['timestamp'][peak])
    print(f"Peak {i + 1}: {peak_time}")
    nearby_gyro_idx = np.where(np.abs(gyropeak_times - peak_time) <= max_diff)[0]

    if len(nearby_gyro_idx) > 0:
        confidence = 100
        source = "both"
    else:
        confidence = calculate_confidence(gyro_magnitude, peak, np.mean(gyro_magnitude), thresholdgyro, local_window_size)
        source = "accel"

    # Append only if confidence is greater than 70
    if confidence > 70:
        diff_from_prev = peak_time - prev_peak_time if prev_peak_time is not None else None
        prev_peak_time = peak_time

        shots.append({
            "shot_number": i + 1,
            "peak_time": round(peak_time, 3),
            "diff_from_prev": round(diff_from_prev, 3) if diff_from_prev is not None else None,
            "confidence": round(confidence, 3),
            "source": source
        })

for i, gyro_peak in enumerate(gyropeaks):
    if not any(np.abs(accelpeak_times - df['timestamp'][gyro_peak]) <= max_diff):  # No matching accel peak
        peak_time = float(df['timestamp'][gyro_peak])

        # Calculate confidence based on local max accel to accel threshold
        confidence = calculate_confidence(resultant_accel, gyro_peak, np.mean(resultant_accel), thresholdaccel, local_window_size)

        # Append only if confidence is greater than 70
        if confidence > 70:
            diff_from_prev = peak_time - prev_peak_time if prev_peak_time is not None else None
            prev_peak_time = peak_time

            shots.append({
                "shot_number": i + 1,
                "peak_time": round(peak_time, 3),
                "diff_from_prev": round(diff_from_prev, 3) if diff_from_prev is not None else None,
                "confidence": round(confidence, 3),
                "source": "gyro"
            })


shots = sorted(shots, key=lambda shot: shot["peak_time"])
for i, shot in enumerate(shots, start=1):
    shot["shot_number"] = i
    if i > 1:
        shot["diff_from_prev"] = round(shot["peak_time"] - prev_peak_time, 3) if prev_peak_time is not None else None
        prev_peak_time = shot["peak_time"]
    else:
        shot["diff_from_prev"] = None
        prev_peak_time = shot["peak_time"]

def segment_data_around_peaks(df, shots, fs=fs):
    """
    Segment the data into strokes around the detected impact peaks using a 1-second window
    (0.5s before and 0.5s after the peak).
    """
    half_window = int(0.5 * fs)  # Convert 0.5 seconds into number of samples
    stroke_data = []

    for shot in shots:
        if not isinstance(shot, dict):  # Ensure shot is a dictionary
            print(f"Skipping invalid shot entry: {shot}")
            continue

        peak_time = shot.get("peak_time")  # Extract peak time in seconds

        if peak_time is None:
            print(f"Warning: Missing peak_time in shot {shot}")  # Debugging
            continue

        peak_idx = df[df['timestamp'] >= peak_time].index[0]



        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(df), peak_idx + half_window)
        if start_idx >= end_idx:
            print(f"Warning: Invalid indices for peak {peak_time} in shot {shot}")
            continue

        stroke_segment = df.iloc[start_idx:end_idx]
        stroke_data.append(stroke_segment)


    return stroke_data

stroke_segments = segment_data_around_peaks(df, shots, fs)
extracted_features = []
to_predict_shot = pd.DataFrame(columns=FEATURE_LABELS)

def extract_features_from_stroke(stroke_segment):
    """
    Extract statistical features from each stroke segment and return as a labeled dictionary.
    """
    row = []
    for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
        row.extend([
            np.mean(stroke_segment[col]), np.std(stroke_segment[col]),
            skew(stroke_segment[col]), kurtosis(stroke_segment[col]),
            np.min(stroke_segment[col]), np.max(stroke_segment[col])
        ])

    return dict(zip(FEATURE_LABELS, row))

for i, segment in enumerate(stroke_segments, start=1):

    to_predict_shot = pd.concat([to_predict_shot, pd.DataFrame([extract_features_from_stroke(segment)])], ignore_index=True)

shot_types = ['serve', 'serve', 'serve', 'forehand', 'backhand', 'serve', 'serve', 'forehand', 'forehand', 'serve', 'serve',
     'forehand', 'forehand', 'serve', 'backhand', 'forehand', 'forehand', 'forehand', 'forehand', 'forehand',
     'forehand', 'forehand', 'forehand', 'forehand', 'forehand', 'backhand']
"""
for i, (segment, shot_type) in enumerate(zip(stroke_segments, shot_types), start=1):

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f'Segmento {i} - Tipo de golpe: {shot_type}')

    # Plot acelerómetro
    axs[0].plot(segment['accelx'], label='accelx')
    axs[0].plot(segment['accely'], label='accely')
    axs[0].plot(segment['accelz'], label='accelz')
    axs[0].set_ylabel('Aceleración (m/s²)')
    axs[0].legend()
    axs[0].grid(True)

    # Giroscopio
    axs[1].plot(segment['gyrox'], label='gyrox')
    axs[1].plot(segment['gyroy'], label='gyroy')
    axs[1].plot(segment['gyroz'], label='gyroz')
    axs[1].set_ylabel('Velocidad Angular (°/s)')
    axs[1].set_xlabel('Muestras')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
"""

csv_filename = "../data/dataset/propio/localdataset.csv"
if not os.path.exists(csv_filename):
    to_predict_shot.to_csv(csv_filename, index_label='Unnamed: 0')

# DESCOMENTAR EN CASO DE AÑADIR

else:
    # Leer el CSV existente para obtener el último índice
    existing_data = pd.read_csv(csv_filename)
    last_index = existing_data["Unnamed: 0"].max() + 1

    # Reasignar índices secuenciales correctos
    to_predict_shot.index = range(last_index, last_index + len(to_predict_shot))

    # Verificar si el archivo termina con un salto de línea
    with open(csv_filename, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        last_char = f.read(1)
        if last_char != b'\n':
            f.write(b'\n')

    # Añadir nuevos datos sin encabezado
    to_predict_shot.to_csv(csv_filename, mode='a', header=False)


folder_name = os.path.basename(os.path.dirname(JSON_FILE))
json_filename = f"../data/detected_shots_{folder_name}.json"
with open(json_filename, "w") as json_file:
    json.dump(shots, json_file, indent=4)

print(f"Detected shots saved to {json_filename}")

data = pd.read_csv("../data/dataset/propio/localdataset.csv")
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=1, stratify=Y
)

def read_video_shots(detected_shots):
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    real_shots = sorted(data, key=lambda x: x["video_time"])
    real_times = [shot["video_time"] for shot in real_shots]
    real_labels = [shot["shot_type"] for shot in real_shots]
    print(f"Presenting sequence of {len(real_shots)} video shots")
    shot_type_map = {"forehand": "for", "backhand": "bck", "serve": "srv"}
    shot_types = [shot_type_map.get(shot["shot_type"], "Unknown") for shot in real_shots]
    print(shot_types)
    return real_shots, real_times, shot_types

# Real shots read from json
real_shots, real_times, real_labels = read_video_shots(detected_shots=shots)
# Detected shots read from json
detected_times = [shot["peak_time"] for shot in shots]

clf = RandomForestClassifier(n_estimators=250,criterion="gini",max_depth=12,min_samples_split=4,min_samples_leaf=1,max_features=8,bootstrap=False,random_state=1)
clf.fit(X_train, Y_train)


def plot_confusion_matrix(Y_test, y_pred, name):
    print(confusion_matrix(Y_test, y_pred))
    matrix = confusion_matrix(Y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = ['Backhand', 'Forehand', 'Serve']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

import numpy as np
from scipy.optimize import linear_sum_assignment

def _robust_offset(real_times, detected_times, rough_window=3.0, fine_window=1.0, fine_step=0.01):
    """
    Estima un desfase global Δ tal que (detected - Δ) ≈ real.
    1) Estimación gruesa: mediana de (nearest_det - real) acotada por rough_window.
    2) Refinamiento: grid search local alrededor de la mediana para minimizar coste L1.
    """
    if len(real_times) == 0 or len(detected_times) == 0:
        return 0.0

    real = np.asarray(real_times, dtype=float)
    det  = np.asarray(detected_times, dtype=float)

    # Paso 1: difs con vecino más cercano
    diffs = []
    j = 0
    for t in real:
        # avanza j para mantenerlo cerca (asume listas ordenadas)
        while j + 1 < len(det) and abs(det[j + 1] - t) <= abs(det[j] - t):
            j += 1
        diffs.append(det[j] - t)
    diffs = np.asarray(diffs)

    # filtra outliers gruesos
    mask = np.abs(diffs) <= rough_window
    if mask.any():
        delta0 = np.median(diffs[mask])
    else:
        delta0 = np.median(diffs)  # por si todo está lejos

    # Paso 2: refinamiento local (L1: suma de mínimos absolutos)
    candidates = np.arange(delta0 - fine_window, delta0 + fine_window + 1e-9, fine_step)
    def l1_cost(delta):
        # Para cada real, coste = distancia al det 'más cercano' una vez desplazados por delta
        det_shift = det - delta
        # dos punteros para O(n+m)
        i = j = 0
        total = 0.0
        while i < len(real) and j < len(det_shift):
            if abs(det_shift[j] - real[i]) <= (abs(det_shift[j-1] - real[i]) if j > 0 else np.inf) and \
               (j+1 == len(det_shift) or abs(det_shift[j] - real[i]) <= abs(det_shift[j+1] - real[i])):
                total += abs(det_shift[j] - real[i])
                i += 1
            else:
                # mover el puntero que más acerque
                if det_shift[j] < real[i]:
                    j += 1
                else:
                    i += 1
        # penaliza no emparejados (aprox): suma de distancias al extremo más cercano
        # (no es crítico; mantiene la función bien comportada)
        return total

    costs = np.array([l1_cost(d) for d in candidates])
    delta_star = candidates[np.argmin(costs)]
    return float(delta_star)

def classify_hungarian_offset(
    clf,
    to_predict_shot,
    real_times,
    detected_times,
    real_labels,
    tolerance: float = 0.5,              # ajusta a tu jitter real
    max_cost_for_dummy: float = 1e6,
    shot_type_map: dict[int, str] = None
):
    """
    1) Estima Δ y desplaza detected_times' = detected_times - Δ.
    2) Aplica método húngaro sobre |det' - real|.
    3) Pares con coste > tolerance -> missing/extra.
    """
    # 0) Clasificar golpes detectados
    y_pred = clf.predict(to_predict_shot) if len(to_predict_shot) else []
    if shot_type_map is None:
        shot_type_map = {0: "srv", 1: "for", 2: "bck"}
    predicted_labels = [shot_type_map.get(int(y), "Unknown") for y in y_pred]

    # Asegura orden temporal
    real_pairs = sorted(zip(real_times, real_labels), key=lambda x: x[0])
    det_pairs  = sorted(zip(detected_times, predicted_labels), key=lambda x: x[0])

    real_times = [t for t, _ in real_pairs]
    real_labels= [l for _, l in real_pairs]
    detected_times = [t for t, _ in det_pairs]
    predicted_labels = [l for _, l in det_pairs]

    # 1) Estimar desfase global Δ
    delta = _robust_offset(real_times, detected_times)
    det_shift = np.array(detected_times, dtype=float) - delta
    real = np.array(real_times, dtype=float)

    n_real = len(real)
    n_det  = len(det_shift)
    size   = max(n_real, n_det)

    # 2) Matriz de costes con padding
    BIG_M = tolerance + max_cost_for_dummy
    cost = np.full((size, size), BIG_M, dtype=float)

    for i in range(n_real):
        for j in range(n_det):
            cost[i, j] = abs(det_shift[j] - real[i])

    row_ind, col_ind = linear_sum_assignment(cost)

    # 3) Construir resultados
    matched_results = []
    used_dets = set()

    for i in range(n_real):
        j = col_ind[i]
        if j < n_det and cost[i, j] <= tolerance:
            used_dets.add(j)
            matched_results.append({
                "status": "matched",
                "real_time": float(real[i]),
                "real_label": real_labels[i],
                "detected_time": float(detected_times[j]),
                "predicted_label": predicted_labels[j],
                "diff": float((det_shift[j] - real[i])),  # ya sin desfase
                "offset": delta
            })
        else:
            matched_results.append({
                "status": "missing",
                "real_time": float(real[i]),
                "real_label": real_labels[i],
                "detected_time": None,
                "predicted_label": None,
                "diff": None,
                "offset": delta
            })

    for j in range(n_det):
        if j not in used_dets:
            matched_results.append({
                "status": "extra",
                "real_time": None,
                "real_label": None,
                "detected_time": float(detected_times[j]),
                "predicted_label": predicted_labels[j],
                "diff": None,
                "offset": delta
            })

    # Ordena por tiempo detectado (None al final)
    matched_results.sort(key=lambda x: x["detected_time"] if x["detected_time"] is not None else float("inf"))

    # Log
    print(f"[INFO] Δ estimado (det - real) = {delta:+.3f}s")
    for k, m in enumerate(matched_results, 1):
        s = m["status"].upper()
        rt = f"{m['real_time']:.3f}" if m["real_time"] is not None else "None"
        dt = f"{m['detected_time']:.3f}" if m["detected_time"] is not None else "None"
        rl = m["real_label"] if m["real_label"] is not None else "---"
        pl = m["predicted_label"] if m["predicted_label"] is not None else "---"
        if s == "MATCHED":
            print(f"Shot {k:02d} | Real: {rl:>3} @ {rt} | Detected: {pl:>3} @ {dt} | DIFF (sin Δ): {m['diff']:+.3f}s | {s}")
        elif s == "MISSING":
            print(f"Shot {k:02d} | Real: {rl:>3} @ {rt} | Detected: --- @ None | {s}")
        else:
            print(f"Shot {k:02d} | Real: --- @ None | Detected: {pl:>3} @ {dt} | {s}")

    # Salidas
    final_labels = []
    for m in matched_results:
        if m["status"] == "missing":
            final_labels.append("M")
        elif m["status"] == "extra":
            final_labels.append("E")
        else:
            final_labels.append(m["predicted_label"])

    matched_real = [m["real_label"] for m in matched_results if m["status"] == "matched"]
    matched_pred = [m["predicted_label"] for m in matched_results if m["status"] == "matched"]

    return final_labels, matched_real, matched_pred, delta



def classify(clf, to_predict_shot, real_times, detected_times):
    # Print test values
    y_pred_test = clf.predict(X_test)
    plot_confusion_matrix(Y_test, y_pred_test, "TEST SVM Lineal")
    print(classification_report(Y_test, y_pred_test))

    y_pred_real = clf.predict(to_predict_shot)
    shot_type_map = {0: "srv", 1: "for", 2: "bck"}
    predicted_labels = [shot_type_map.get(shot, "Unknown") for shot in y_pred_real]

    tolerance = 1.5  # seconds
    matched_results = []
    used_detected = set()

    reference_diff = None

    for i, real_time in enumerate(real_times):
        closest = None
        min_diff = float("inf")

        for j, det_time in enumerate(detected_times):
            if j in used_detected:
                continue

            diff = det_time - real_time  # signed difference

            if i == 0 and j == 0:
                reference_diff = diff  # save the initial diff as reference
                closest = j
                break  # no need to keep searching
            elif reference_diff is not None and abs(diff - reference_diff) <= tolerance:
                if abs(diff - reference_diff) < min_diff:
                    min_diff = abs(diff - reference_diff)
                    closest = j

        if closest is not None:
            used_detected.add(closest)
            matched_results.append({
                "real_time": real_time,
                "real_label": real_labels[i],
                "detected_time": detected_times[closest],
                "predicted_label": predicted_labels[closest],
                "index": closest,
                "status": "matched"
            })
        else:
            matched_results.append({
                "real_time": real_time,
                "real_label": real_labels[i],
                "detected_time": None,
                "predicted_label": None,
                "status": "missing"
            })

    # Add false positives (unmatched detections)
    for j, det_time in enumerate(detected_times):
        if j not in used_detected:
            matched_results.append({
                "real_time": None,
                "real_label": None,
                "detected_time": det_time,
                "predicted_label": predicted_labels[j],
                "index": j,
                "status": "extra"
            })

    matched_results.sort(key=lambda x: (
        x["detected_time"] if x["detected_time"] is not None else float("inf")
    ))

    for i, match in enumerate(matched_results, start=1):
        status = match["status"]
        r_label = match["real_label"] or "---"
        p_label = match["predicted_label"] or "---"
        real_time = match["real_time"]
        detected_time = match["detected_time"]

        if status == "matched" and detected_time is not None and real_time is not None:
            actual_diff = round(detected_time - real_time, 3)
            print(
                f"Shot {i:02d} | Real: {r_label:>3} @ {real_time:.2f} | Detected: {p_label:>3} @ {detected_time:.3f} | DIFF: {actual_diff:+.3f}s | REF: {reference_diff:+.3f}s | {status.upper()}")
        elif status == "missing":
            print(f"Shot {i:02d} | Real: {r_label:>3} @ {real_time:.2f} | Detected: --- @ None | {status.upper()}")
        elif status == "extra":
            print(f"Shot {i:02d} | Real: --- @ None | Detected: {p_label:>3} @ {detected_time:.3f} | {status.upper()}")

    final_labels = []
    for match in matched_results:
        if match["status"] in ["missing"]:
            final_labels.append("M")
        elif match["status"] in ["extra"]:
            final_labels.append("E")
        else:
            final_labels.append(match["predicted_label"])  # already human-readable

    matched_real = [m["real_label"] for m in matched_results if m["status"] == "matched"]
    matched_pred = [m["predicted_label"] for m in matched_results if m["status"] == "matched"]

    return final_labels, matched_real, matched_pred

final_labels, matched_real, matched_pred = classify(clf, to_predict_shot, real_times, detected_times)
"""
final_labels, matched_real, matched_pred, delta = classify_hungarian_offset(
    clf, to_predict_shot, real_times, detected_times, real_labels, tolerance=1.5
)
"""
print("Random Forest :")
print(f"Presenting sequence of {len(final_labels)} shots")
print(final_labels)
plot_confusion_matrix(matched_real, matched_pred, "Random Forest")
print(classification_report(matched_real,matched_pred ))

print("SVM Lineal:")
clf = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=False) if hasattr(X, "sparse") and X.sparse else StandardScaler()),
    ("svm", LinearSVC(C=1.0, random_state=1, max_iter=10000))
])
clf.fit(X_train, Y_train)
os.makedirs("../pc_server/models", exist_ok=True)
joblib.dump(clf, "../pc_server/models/rf_model.pkl")
print("✅ Modelo guardado en ./models/rf_model.pkl")
lsvm_final_labels, lsvm_matched_real, lsvm_matched_pred = classify(clf, to_predict_shot, real_times, detected_times)
print(f"Presenting sequence of {len(lsvm_final_labels)} shots")
print(lsvm_final_labels)
plot_confusion_matrix(lsvm_matched_real, lsvm_matched_pred, "SVM Lineal")
print(classification_report(lsvm_matched_real,lsvm_matched_pred ))


"""
print("SVM Radial:")
clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=1)
clf.fit(X_train, Y_train)
svc_final_labels, svc_matched_real, svc_matched_pred = classify(clf, to_predict_shot, real_times, detected_times)
print(f"Presenting sequence of {len(svc_final_labels)} shots")
print(svc_final_labels)
plot_confusion_matrix(svc_matched_real, svc_matched_pred, "SVM Radial")
print(classification_report(svc_matched_real,svc_matched_pred ))

print("KNN")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
knn = KNeighborsClassifier(n_neighbors=7)  # Try 3, 5, 7, etc.
knn.fit(X_train_scaled, Y_train)
knn_final_labels, knn_matched_real, knn_matched_pred = classify(knn, to_predict_shot, real_times, detected_times)
print(f"Presenting sequence of {len(knn_final_labels)} shots")
print(knn_final_labels)
plot_confusion_matrix(knn_matched_real, knn_matched_pred, "KNN")
print(classification_report(knn_matched_real,knn_matched_pred,zero_division=0))

print("HMM")
n_classes = len(np.unique(Y_train))
model = hmm.GaussianHMM(n_components=n_classes, covariance_type='full', n_iter=500)
model.fit(X_train_scaled)
hmm_final_labels, hmm_matched_real, hmm_matched_pred = classify(model, to_predict_shot, real_times, detected_times)
print(f"Presenting sequence of {len(hmm_final_labels)} shots")
print(hmm_final_labels)
plot_confusion_matrix(hmm_matched_real, hmm_matched_pred, "HMM")
print(classification_report(hmm_matched_real,hmm_matched_pred,zero_division=0))
"""

