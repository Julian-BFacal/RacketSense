import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# Load CSV data
csv_filename = '../data/dentro1.csv'
df = pd.read_csv(csv_filename)
#13-03-25_15;51_amarilla 76,178
#13-03-25_16;00_amarilla 25,130
#13-03-25_15;51_azul 65,165 
#13-03-25_16;00_azul 32,127
xllim, xhlim = 0, 150

# Extraer solo el nombre del archivo sin la r   uta
title_name = csv_filename.split('/')[-1]


#https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
def mean_timestamp_difference(csv_file):
    # Cargar el archivo CSV en un DataFrame

    # Calcular las diferencias entre timestamps consecutivos
    df['timestamp_diff'] = df['timestamp'].diff()

    # Calcular la media de las diferencias (ignorando el primer NaN)
    mean_diff = df['timestamp_diff'].mean()

    return mean_diff

print(mean_timestamp_difference(df))
fs = 1 / mean_timestamp_difference(df)
print(fs)

# Ensure required columns exist
required_columns = {'timestamp', 'accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file is missing required columns: {required_columns - set(df.columns)}")
# Plot raw IMU Data
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
fig.suptitle(f'Datos IMU - {title_name}', fontsize=14)  # Título global
axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox', marker='x' )
axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy', marker='x')
axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz', marker='x')
axes[0].set_title('Raw Gyroscope Data')
axes[0].legend()
axes[0].grid()
axes[0].set_xlim(xllim, xhlim)

axes[1].plot(df['timestamp'], df['accelx'], label='accelx',marker='x')
axes[1].plot(df['timestamp'], df['accely'], label='accely',marker='x')
axes[1].plot(df['timestamp'], df['accelz'], label='accelz',marker='x')
axes[1].set_title('Raw Accelerometer Data')
axes[1].legend()
axes[1].grid()
axes[1].set_xlim(xllim, xhlim)
plt.tight_layout()
plt.show(block=False)

raw_resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
raw_gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)

min_distance_samples = int(0.5 * fs)

threshold = np.max(raw_resultant_accel) * 0.35 # 20% threshold
thresholdgyro = np.max(raw_gyro_magnitude) * 0.35 # 20% threshold

AccX_peaks, _ = find_peaks(raw_resultant_accel, height=threshold, distance=min_distance_samples)
Gyro_peaks, _ = find_peaks(raw_gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)

# Plot raw magnitude of IMU Data
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
fig.suptitle(f'Raw Magnitude of IMU Data - {title_name}', fontsize=14)

axes[0].plot(df['timestamp'], raw_gyro_magnitude, label='Raw Gyro Magnitude')
axes[0].set_title('Raw Gyroscope Magnitude')
axes[0].legend()
axes[0].grid()
axes[0].set_xlim(xllim, xhlim)
axes[0].axhline(np.max(raw_gyro_magnitude), color='red', linestyle='--', label='Maximum')
axes[0].axhline(thresholdgyro, color='green', linestyle='-.', label='Seuil de detection')
axes[0].axhline(np.mean(raw_gyro_magnitude), color='yellow', linestyle='-.', label='Seuil de detection')


for peak in Gyro_peaks:
    axes[0].axvline(x=df['timestamp'].iloc[peak], color='r', linestyle='--', label='Peak')

axes[1].plot(df['timestamp'], raw_resultant_accel, label='Raw Accel Magnitude')
axes[1].set_title('Raw Accelerometer Magnitude')
axes[1].legend()
axes[1].grid()
axes[1].set_xlim(xllim, xhlim)
axes[1].axhline(np.max(raw_resultant_accel), color='red', linestyle='--', label='Maximum')
axes[1].axhline(threshold, color='green', linestyle='-.', label='Seuil de detection')
axes[1].axhline(np.mean(raw_resultant_accel), color='yellow', linestyle='-.', label='Seuil de detection')


for peak in AccX_peaks:
    axes[1].axvline(x=df['timestamp'].iloc[peak], color='r', linestyle='--', label='Peak')

"""
# Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff=10, fs=30, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply Low-Pass Filter and Zero-Phase Filtering to Acceleration and Gyroscope Data
df['accelx'] = butter_lowpass_filter(df['accelx'])
df['accely'] = butter_lowpass_filter(df['accely'])
df['accelz'] = butter_lowpass_filter(df['accelz'])
df['gyrox'] = butter_lowpass_filter(df['gyrox'])
df['gyroy'] = butter_lowpass_filter(df['gyroy'])
df['gyroz'] = butter_lowpass_filter(df['gyroz'])



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
df['accelx'] = wavelet_denoise(df['accelx'])
df['accely'] = wavelet_denoise(df['accely'])
df['accelz'] = wavelet_denoise(df['accelz'])
df['gyrox'] = wavelet_denoise(df['gyrox'])
df['gyroy'] = wavelet_denoise(df['gyroy'])
df['gyroz'] = wavelet_denoise(df['gyroz'])
"""

# Plot filtered IMU Data
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)


axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox')
axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy')
axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz')
axes[0].set_title('Filtered Gyroscope Data')
axes[0].legend()
axes[0].grid()
axes[0].set_xlim(xllim, xhlim)

axes[1].plot(df['timestamp'], df['accelx'], label='accelx')
axes[1].plot(df['timestamp'], df['accely'], label='accely')
axes[1].plot(df['timestamp'], df['accelz'], label='accelz')
axes[1].set_title('Filtered Accelerometer Data')
axes[1].legend()
axes[1].grid()
axes[1].set_xlim(xllim, xhlim)
plt.tight_layout()
plt.show(block=False)

# Compute resultant acceleration and gyroscope magnitude
resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)

# Peak Detection
min_distance_samples = int(0.5 * fs)

threshold = np.max(resultant_accel) * 0.35 # 20% threshold
thresholdgyro = np.max(gyro_magnitude) * 0.35 # 20% threshold

AccX_peaks, _ = find_peaks(resultant_accel, height=threshold, distance=min_distance_samples)
Gyro_peaks, _ = find_peaks(gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)

print("mean",np.mean(gyro_magnitude),np.mean(resultant_accel))


# Plot resultant acceleration and gyroscope magnitude
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axes[0].plot(df['timestamp'], gyro_magnitude, label='gyro magnitude')
axes[1].set_title('gyro magnitude')
axes[0].legend()
axes[0].grid()
axes[0].set_xlim(xllim, xhlim)
axes[0].axhline(np.max(gyro_magnitude), color='red', linestyle='--', label='Maximum')
axes[0].axhline(thresholdgyro, color='green', linestyle='-.', label='Seuil de detection')
axes[0].axhline(np.mean(gyro_magnitude), color='yellow', linestyle='-.', label='Seuil de detection')


for peak in Gyro_peaks:
    axes[0].axvline(x=df['timestamp'].iloc[peak], color='r', linestyle='--', label='Peak')


axes[1].plot(df['timestamp'], resultant_accel, label='accel magnitude')
axes[1].set_title('Accelerometer Data')
axes[1].legend()
axes[1].grid()
axes[1].set_xlim(xllim, xhlim)

axes[1].axhline(np.max(resultant_accel), color='red', linestyle='--', label='Maximum')
axes[1].axhline(threshold, color='green', linestyle='-.', label='Seuil de detection')
axes[1].axhline(np.mean(resultant_accel), color='yellow', linestyle='-.', label='Seuil de detection')

for peak in AccX_peaks:
    axes[1].axvline(x=df['timestamp'].iloc[peak], color='r', linestyle='--', label='Peak')

plt.tight_layout()
plt.show(block=True)


# Feature Extraction
cols_labels = [
    "AccXMean", "AccXSD", "AccXSkew", "AccXKurtosis", "AccXMin", "AccXMax",
    "AccYMean", "AccYSD", "AccYSkew", "AccYKurtosis", "AccYMin", "AccYMax",
    "AccZMean", "AccZSD", "AccZSkew", "AccZKurtosis", "AccZMin", "AccZMax",
    "GyrXMean", "GyrXSD", "GyrXSkew", "GyrXKurtosis", "GyrXMin", "GyrXMax",
    "GyrYMean", "GyrYSD", "GyrYSkew", "GyrYKurtosis", "GyrYMin", "GyrYMax",
    "GyrZMean", "GyrZSD", "GyrZSkew", "GyrZKurtosis", "GyrZMin", "GyrZMax"
]
to_predict_shot = pd.DataFrame(columns=cols_labels)

for i in AccX_peaks:
    interval = 30  # Adjusted interval for windowing
    data_segment = df.iloc[max(0, i-interval):min(len(df), i+interval)]
    row = []
    for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
        row.extend([
            np.mean(data_segment[col]), np.std(data_segment[col]),
            skew(data_segment[col]), kurtosis(data_segment[col]),
            np.min(data_segment[col]), np.max(data_segment[col])
        ])
    to_predict_shot.loc[len(to_predict_shot)] = row
