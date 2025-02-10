from cmath import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# Use TkAgg backend for interactive zoom/pan (works best in PyCharm)
matplotlib.use('TkAgg')

# Load the CSV data
df = pd.read_csv('imu_data.csv')
resultant_accel = np.sqrt(df["accelx"]**2 + df["accely"]**2 + df["accelz"]**2)
# Calculate Gyroscope Magnitude
gyro_magnitude = np.sqrt(df["gyrox"]**2 + df["gyroy"]**2 + df["gyroz"]**2)

# Ensure timestamp exists
if 'timestamp' not in df.columns:
    raise ValueError("CSV file is missing the 'timestamp' column.")

# Enable interactive mode
plt.ion()

# Create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot Gyroscope Data
axes[0].plot(df['timestamp'], df['gyrox'], label='gyrox', marker='x')
axes[0].plot(df['timestamp'], df['gyroy'], label='gyroy', marker='x')
axes[0].plot(df['timestamp'], df['gyroz'], label='gyroz', marker='x')
axes[0].set_ylabel('Gyroscope Values')
axes[0].set_title('IMU Gyroscope Data')
axes[0].legend()
axes[0].grid()

# Plot Accelerometer Data
axes[1].plot(df['timestamp'], df['accelx']/16384, label='accelx', marker='x')
axes[1].plot(df['timestamp'], df['accely']/16384, label='accely', marker='x')
axes[1].plot(df['timestamp'], df['accelz']/16384, label='accelz', marker='x')
print()
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Accelerometer Values')
axes[1].set_title('IMU Accelerometer Data')
axes[1].legend()
axes[1].grid()

# Adjust layout
plt.tight_layout()
plt.show(block=False)

# Create figure with two subplots for the new figure
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
# Plot Resultant Acceleration

# Plot Gyroscope Magnitude (on top)
axes2[0].plot(df['timestamp'], gyro_magnitude, label='Gyroscope Magnitude', color='g')
axes2[0].set_ylabel('Gyroscope Magnitude')
axes2[0].set_title('IMU Gyroscope Magnitude')
axes2[0].legend()
axes2[0].grid()

# Plot Accelerometer Magnitude (on bottom)
axes2[1].plot(df['timestamp'], resultant_accel, label='Accelerometer Magnitude', color='r')
axes2[1].set_xlabel('Time (seconds)')
axes2[1].set_ylabel('Accelerometer Magnitude')
axes2[1].set_title('IMU Accelerometer Magnitude')
axes2[1].legend()
axes2[1].grid()

# Show the second figure
plt.show(block=True)

# Show the plot in a separate interactive window
plt.show(block=True)

# Adjust layout
plt.tight_layout()

# Show the new figure
plt.show(block=True)


