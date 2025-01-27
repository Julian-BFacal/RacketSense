import asyncio
import csv
import os
import keyboard  # Requires `pip install keyboard`
from bleak import BleakClient, BleakScanner

# UUIDs for the service and characteristics
SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
ACCEL_CHARACTERISTIC_UUID = "2C06"
GYRO_CHARACTERISTIC_UUID = "2C09"

# Ensure 'data' folder exists
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(data_folder, exist_ok=True)

# CSV file setup (overwrite existing file)
csv_filename = os.path.join(data_folder, "imu_data.csv")

# Write header (overwrite file) at the start of the program
with open(csv_filename, mode="w", newline="") as csv_file:
    fieldnames = ['timestamp', 'gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

# Flag to track when to stop
stop_flag = False

# Data storage to accumulate readings for one row
data_row = {'timestamp': '', 'gyrox': '', 'gyroy': '', 'gyroz': '', 'accelx': '', 'accely': '', 'accelz': ''}

# Function to detect spacebar press
def wait_for_spacebar():
    global stop_flag
    print("[INFO] Press SPACEBAR to stop recording.")
    keyboard.wait("space")
    stop_flag = True
    print("\n[INFO] Spacebar pressed. Stopping data collection...")

# Function to find the IMU device dynamically
async def find_device():
    while True:
        print("Scanning for devices...")

        devices = await BleakScanner.discover(return_adv=True)

        for device, advertisement_data in devices.values():
            if SERVICE_UUID.lower() in [s.lower() for s in advertisement_data.service_uuids]:
                print(f"Found device: {device.name} ({device.address}) with expected service UUID.")
                return device.address  # Stop looping once the device is found

        print("No matching device found. Retrying...\n")
        await asyncio.sleep(5)  # Wait before retrying

# Function to handle accelerometer data
def accel_handler(sender, data):
    if stop_flag:
        return  # Stop processing data if spacebar was pressed

    try:
        accelx, accely, accelz, timestamp = map(float, data.decode('utf-8').split(','))
        print(f"Accel: x={accelx}, y={accely}, z={accelz}, time={timestamp}")

        # Update the data_row with accelerometer values
        data_row['accelx'] = accelx
        data_row['accely'] = accely
        data_row['accelz'] = accelz
        data_row['timestamp'] = (timestamp)/1000  # Ensure timestamp is an integer

        # Only write the row when both accelerometer and gyroscope data are received
        if data_row['gyrox'] != '':
            write_data_row()
    except Exception as e:
        print(f"Error decoding accelerometer data: {e}")

# Function to handle gyroscope data
def gyro_handler(sender, data):
    if stop_flag:
        return  # Stop processing data if spacebar was pressed

    try:
        gyrox, gyroy, gyroz, timestamp = map(float, data.decode('utf-8').split(','))
        print(f"Gyro: x={gyrox}, y={gyroy}, z={gyroz}, time={timestamp}")

        # Update the data_row with gyroscope values
        data_row['gyrox'] = gyrox
        data_row['gyroy'] = gyroy
        data_row['gyroz'] = gyroz
        data_row['timestamp'] = (timestamp)/1000  # Ensure timestamp is an integer

        # Only write the row when both accelerometer and gyroscope data are received
        if data_row['accelx'] != '':
            write_data_row()
    except Exception as e:
        print(f"Error decoding gyroscope data: {e}")

def write_data_row():
    global data_row
    with open(csv_filename, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['timestamp', 'gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz'])
        writer.writerow(data_row)
        print(f"[DATA SAVED] {data_row}")

    # Reset the row for the next data entry
    reset_data_row()

# Function to reset the data row for the next entry
def reset_data_row():
    global data_row
    data_row = {'timestamp': '', 'gyrox': '', 'gyroy': '', 'gyroz': '', 'accelx': '', 'accely': '', 'accelz': ''}

# Main function to connect and read data
async def main():
    global stop_flag

    # Start key listener in a separate thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, wait_for_spacebar)

    device_address = await find_device()
    if not device_address:
        print("No suitable device found. Exiting.")
        return

    print(f"Connecting to {device_address}...")
    async with BleakClient(device_address, timeout=60) as client:
        print("Connected!")

        # Subscribe to accelerometer characteristic
        await client.start_notify(ACCEL_CHARACTERISTIC_UUID, accel_handler)

        # Subscribe to gyroscope characteristic
        await client.start_notify(GYRO_CHARACTERISTIC_UUID, gyro_handler)

        print("[INFO] Listening for data...")

        # Keep looping until spacebar is pressed
        while not stop_flag:
            await asyncio.sleep(0.5)

        print("[INFO] Stopping data collection and saving CSV.")

if __name__ == "__main__": v
    asyncio.run(main())
