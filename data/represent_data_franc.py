import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# Load CSV data
df = pd.read_csv('DatosJorge/saques.csv')

# Ensure required columns exist
required_columns = {'timestamp', 'accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file is missing required columns: {required_columns - set(df.columns)}")

# Plot raw IMU Data
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox', marker='x')
axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy', marker='x')
axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz', marker='x')
axes[0].set_title('Raw Gyroscope Data')
axes[0].legend()
axes[0].grid()

axes[1].plot(df['timestamp'], df['accelx'], label='accelx', marker='x')
axes[1].plot(df['timestamp'], df['accely'], label='accely', marker='x')
axes[1].plot(df['timestamp'], df['accelz'], label='accelz', marker='x')
axes[1].set_title('Raw Accelerometer Data')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.show(block=False)

# Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff=40, fs=560, order=3):
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

# Plot filtered IMU Data
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox')
axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy')
axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz')
axes[0].set_title('Filtered Gyroscope Data')
axes[0].legend()
axes[0].grid()

axes[1].plot(df['timestamp'], df['accelx'], label='accelx')
axes[1].plot(df['timestamp'], df['accely'], label='accely')
axes[1].plot(df['timestamp'], df['accelz'], label='accelz')
axes[1].set_title('Filtered Accelerometer Data')
axes[1].legend()
axes[1].grid()
plt.tight_layout()
plt.show(block=False)

# Compute resultant acceleration and gyroscope magnitude
resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)

# Peak Detection
print(np.max(resultant_accel))

threshold = np.max(resultant_accel) * 0.4 # 20% threshold
AccX_peaks, _ = find_peaks(resultant_accel, height=threshold, distance=10)



# Plot resultant acceleration and gyroscope magnitude
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axes[0].plot(df['timestamp'], gyro_magnitude, label='gyro magnitude')
axes[1].set_title('gyro magnitude')
axes[0].legend()
axes[0].grid()
axes[1].plot(df['timestamp'], resultant_accel, label='accel magnitude')
axes[1].set_title('Accelerometer Data')
axes[1].legend()
axes[1].grid()

axes[1].axhline(np.max(resultant_accel), color='red', linestyle='--', label='Maximum')
axes[1].axhline(threshold, color='green', linestyle='-.', label='Seuil de detection')

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

"""# Load Training Data
data = pd.read_csv("dataset/training_dataset.csv")
data.drop(columns=["Unnamed: 0"], axis=1, inplace=True)
X = data.loc[:, data.columns != "TypeOfShot"]
Y = data["TypeOfShot"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train and Evaluate Random Forest
clf_rf = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_split=4, max_features=8)
clf_rf.fit(X_train, Y_train)
y_pred_rf = clf_rf.predict(X_test)
print("Random Forest Prediction:", clf_rf.predict(to_predict_shot))
print("Accuracy:", accuracy_score(Y_test, y_pred_rf))
print("Precision:", precision_score(Y_test, y_pred_rf, average='weighted', zero_division=1))

# Train and Evaluate SVM
clf_svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
clf_svm.fit(X_train, Y_train)
y_pred_svm = clf_svm.predict(X_test)
print("SVM Prediction:", clf_svm.predict(to_predict_shot))
print("Accuracy:", accuracy_score(Y_test, y_pred_svm))
print("Precision:", precision_score(Y_test, y_pred_svm, average='weighted', zero_division=1))
"""