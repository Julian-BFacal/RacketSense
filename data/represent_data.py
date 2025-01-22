import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Use TkAgg backend for interactive zoom/pan (works best in PyCharm)
matplotlib.use('TkAgg')

# Load the CSV data
df = pd.read_csv('imu_data.csv')

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
axes[1].plot(df['timestamp'], df['accelx'], label='accelx', marker='x')
axes[1].plot(df['timestamp'], df['accely'], label='accely', marker='x')
axes[1].plot(df['timestamp'], df['accelz'], label='accelz', marker='x')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Accelerometer Values')
axes[1].set_title('IMU Accelerometer Data')
axes[1].legend()
axes[1].grid()

# Adjust layout
plt.tight_layout()

# Show the plot in a separate interactive window
plt.show(block=True)
