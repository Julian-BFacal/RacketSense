import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt
from scipy.stats import kurtosis, skew

# Load CSV data
csv_filename = '../data/SensorTriangulo/26-04-25_10;35/26-04-25_10;35.csv'
df = pd.read_csv(csv_filename)
x_min, x_max = 0, 250
df = df[(df['timestamp'] >= x_min) & (df['timestamp'] <= x_max)]
df.reset_index(drop=True, inplace=True)

#13-03-25_15;51_amarilla 76,178
#13-03-25_16;00_amarilla 0,130
#13-03-25_15;51_azul 65,165
#13-03-25_16;00_azul 32,127
#26-04-25_10;15 0,250

title_name = csv_filename.split('_')[-1]


def mean_timestamp_difference(csv_file):
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

raw_resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
raw_gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)


min_distance_samples = int(0.5 * fs)


def plot_raw_data():
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'Datos IMU - {title_name}', fontsize=14)  # Título global
    axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox')
    axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy')
    axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz')
    axes[0].set_title('Raw Gyroscope Data')
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlim(x_min, x_max)

    axes[1].plot(df['timestamp'], df['accelx'], label='accelx')
    axes[1].plot(df['timestamp'], df['accely'], label='accely')
    axes[1].plot(df['timestamp'], df['accelz'], label='accelz')
    axes[1].set_title('Raw Accelerometer Data')
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlim(x_min, x_max)
    plt.tight_layout()
    plt.show(block=False)



def plot_magnitudes(gyro_magnitude, accel_magnnitude, title, block, number):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    fig.suptitle(f"{csv_filename.split('_')[1]} - Sensor Data", fontsize=14, fontweight='bold')

    # Set x-axis limits
    threshold = np.max(accel_magnnitude) * 0.35  # 20% threshold
    thresholdgyro = np.max(gyro_magnitude) * 0.35  # 20% threshold
    AccX_peaks, _ = find_peaks(accel_magnnitude, height=threshold, distance=min_distance_samples)
    Gyro_peaks, _ = find_peaks(gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)


    # Gyroscope plot
    axes[0].plot(df['timestamp'], gyro_magnitude, label='Gyro Magnitude')
    axes[0].set_title(f"{title} Gyro Magnitude")
    axes[0].grid()
    #axes[0].set_xlim(x_min, x_max)

    # Annotate important values
    max_gyro = np.max(gyro_magnitude)
    mean_gyro = np.mean(gyro_magnitude)

    axes[0].axhline(max_gyro, color='black', linestyle='--', label='Maximum')
    axes[0].axhline(thresholdgyro, color='green', linestyle='-.', label='Threshold 0.35xMaximum')
    axes[0].axhline(mean_gyro, color='yellow', linestyle='-.', label='Mean Gyro Magnitude')

    # Define label x-position (shifted right into the white space)
    label_x_pos = x_max + (x_max - x_min) * 0.02  # 2% beyond x_max

    # Add text labels next to each line in the white space
    axes[0].text(label_x_pos, max_gyro, f'Max Gyro: {max_gyro:.2f}', color='black', fontsize=10, verticalalignment='bottom')
    axes[0].text(label_x_pos, thresholdgyro, f'Threshold Gyro: {thresholdgyro:.2f}', color='green', fontsize=10, verticalalignment='bottom')
    axes[0].text(label_x_pos, mean_gyro, f'Mean Gyro: {mean_gyro:.2f}', color='yellow', fontsize=10, verticalalignment='bottom')

    for i, peak in enumerate(Gyro_peaks, start=1):  # Start numbering from 1
        axes[0].axvline(x=df['timestamp'][peak], color='r', linestyle='--')
        axes[0].text(df['timestamp'][peak], gyro_magnitude[peak], str(i), color='red', fontsize=9,
                     verticalalignment='bottom')

    # Accelerometer plot
    axes[1].plot(df['timestamp'], accel_magnnitude, label='Accel Magnitude')
    axes[1].set_title(f"{title} Accelerometer Magnitude")
    axes[1].grid()
    #axes[1].set_xlim(x_min, x_max)

    max_accel = np.max(accel_magnnitude)
    mean_accel = np.mean(accel_magnnitude)

    axes[1].axhline(max_accel, color='red', linestyle='--', label='Maximum')
    axes[1].axhline(threshold, color='green', linestyle='-.', label='Threshold 0.35xMaximum')
    axes[1].axhline(mean_accel, color='yellow', linestyle='-.', label='Mean Accel Magnitude')

    # Add text labels for accelerometer in the white space
    axes[1].text(label_x_pos, max_accel, f'Max Accel: {max_accel:.2f}', color='red', fontsize=10,
                 verticalalignment='bottom')
    axes[1].text(label_x_pos, threshold, f'Threshold: {threshold:.2f}', color='green', fontsize=10, verticalalignment='bottom')
    axes[1].text(label_x_pos, mean_accel, f'Mean Accel: {mean_accel:.2f}', color='yellow', fontsize=10, verticalalignment='bottom')

    for i, peak in enumerate(AccX_peaks, start=1):
        axes[1].axvline(x=df['timestamp'].iloc[peak], color='r', linestyle='--')
        axes[1].text(df['timestamp'][peak], accel_magnnitude[peak], str(i), color='red', fontsize=9,
                     verticalalignment='bottom')

    plt.tight_layout()

    # Save the figure with the title as filename (replacing spaces with underscores)
    filename = f"{number}{title.replace(' ', '')}_magnitudes.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show(block=block)

plot_raw_data()

# Print raw magnitudes with shot detection
plot_magnitudes(raw_gyro_magnitude, raw_resultant_accel, "Raw ", False, "1.")

# Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff=6, fs=60, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Compute resultant acceleration and gyroscope magnitude after Low-Pass Filter
butter_accel_mag = np.sqrt(butter_lowpass_filter(df['accelx'])**2 + butter_lowpass_filter(df['accely'])**2 + butter_lowpass_filter(df['accelz'])**2)
butter_gyro_mag = np.sqrt(butter_lowpass_filter(df['gyrox'])**2 + butter_lowpass_filter(df['gyroy'])**2 + butter_lowpass_filter(df['gyroz'])**2)

# Print magnitudes with shot detection
plot_magnitudes(butter_gyro_mag, butter_accel_mag, "Butterworth Low-Pass Filtered ", False, "2.")

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

# Compute resultant acceleration and gyroscope magnitude after Low-Pass Filter
wavelet_accel_mag = np.sqrt(wavelet_denoise(df['accelx'])**2 + wavelet_denoise(df['accely'])**2 + wavelet_denoise(df['accelz'])**2)
wavelet_gyro_mag = np.sqrt(wavelet_denoise(df['gyrox'])**2 + wavelet_denoise(df['gyroy'])**2 + wavelet_denoise(df['gyroz'])**2)


# Print magnitudes with shot detection
plot_magnitudes(wavelet_gyro_mag, wavelet_accel_mag, "Wavelet Denoised Filtered ", False, "3.")


for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
    df[col] = butter_lowpass_filter(df[col])
for col in ['accelx', 'accely', 'accelz', 'gyrox', 'gyroy', 'gyroz']:
    df[col] = wavelet_denoise(df[col])
# Compute resultant acceleration and gyroscope magnitude
resultant_accel = np.sqrt(df['accelx']**2 + df['accely']**2 + df['accelz']**2)
gyro_magnitude = np.sqrt(df['gyrox']**2 + df['gyroy']**2 + df['gyroz']**2)


plot_magnitudes(gyro_magnitude, resultant_accel, "Both Filtered ", True, "4.")

# Peak Detection
min_distance_samples = int(0.5 * fs)

threshold = np.max(resultant_accel) * 0.35 # 20% threshold
thresholdgyro = np.max(gyro_magnitude) * 0.35 # 20% threshold

print(threshold, thresholdgyro)

AccX_peaks, _ = find_peaks(resultant_accel, height=threshold, distance=min_distance_samples)
Gyro_peaks, _ = find_peaks(gyro_magnitude, height=thresholdgyro, distance=min_distance_samples)

print("mean",np.mean(gyro_magnitude),np.mean(resultant_accel))


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