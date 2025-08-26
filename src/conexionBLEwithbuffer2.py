
import asyncio
import csv
import os
import queue
import threading
import keyboard  # Requires pip install keyboard
from bleak import BleakClient, BleakScanner
import struct

# UUIDs for the service and characteristics
SERVICE_UUID = "12345678-1234-1234-1234-123456789def"
BLE_CHARACTERISTIC_UUID = "2C06"

# Ensure 'data' folder  nexists
data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(data_folder, exist_ok=True)

# CSV file setup
csv_filename = os.path.join(data_folder, "imu_data2.csv")

# Create a queue for buffering data before writing  
data_queue = queue.Queue()

# Flag to stop recording
stop_flag = False

# Data storage
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

def imu_handler(sender, data):
    if stop_flag:
        return
    try:
        ax, ay, az, gx, gy, gz, timestamp = struct.unpack("iiiiiii", data)
        ax /= 10000.0
        ay /= 10000.0
        az /= 10000.0
        gx /= 10000.0
        gy /= 10000.0
        gz /= 10000.0

        print(f"Accel: {ax}, {ay}, {az} | Gyro: {gx}, {gy}, {gz} | Time: {timestamp}")

        data_row['gyrox'] = gx
        data_row['gyroy'] = gy
        data_row['gyroz'] = gz
        data_row['accelx'] = ax
        data_row['accely'] = ay
        data_row['accelz'] =  az
        data_row['timestamp'] = timestamp / 1000
        data_queue.put(data_row.copy())

    except struct.error as e:
        print(f"Error unpacking gyroscope data: {e}")


"""
def imu_handler(sender, data):
    if stop_flag:
        return
    try:
        # Ensure data is a multiple of 28 bytes
        num_readings = len(data) // 28  # How many readings are in this batch?

        for i in range(num_readings):
            offset = i * 28  # Move through the buffer for each reading
            ax, ay, az, gx, gy, gz, timestamp = struct.unpack("iiiiiii", data[offset:offset+28])

            ax /= 10000.0
            ay /= 10000.0
            az /= 10000.0
            gx /= 10000.0
            gy /= 10000.0
            gz /= 10000.0 

            print(f"Accel: {ax}, {ay}, {az} | Gyro: {gx}, {gy}, {gz} | Time: {timestamp}")

            data_row['gyrox'] = gx
            data_row['gyroy'] = gy
            data_row['gyroz'] = gz
            data_row['accelx'] = ax
            data_row['accely'] = ay
            data_row['accelz'] = az
            data_row['timestamp'] = timestamp / 1000

            data_queue.put(data_row.copy())

    except struct.error as e:
        print(f"Error unpacking data: {e}")
    """


# Function to reset the data row for the next entry
def reset_data_row():
    global data_row
    data_row = {'timestamp': '', 'gyrox': '', 'gyroy': '', 'gyroz': '', 'accelx': '', 'accely': '', 'accelz': ''}

# Writer thread function
def write_data_thread():
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ['timestamp', 'gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while not stop_flag or not data_queue.empty():
            try:
                row = data_queue.get(timeout=1)
                writer.writerow(row)
                csv_file.flush()
                print(f"[DATA SAVED] {row}")
            except queue.Empty:
                pass  # No new data, keep looping

    print("[INFO] Writer thread finished writing all data.")

# Main function to connect and read data
async def main():
    global stop_flag

    # Start key listener in a separate thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, wait_for_spacebar)

    # Start writer thread
    writer_thread = threading.Thread(target=write_data_thread, daemon=True)
    writer_thread.start()

    device_address = await find_device()
    if not device_address:
        print("No suitable device found. Exiting.")
        return

    print(f"Connecting to {device_address}...")
    async with BleakClient(device_address, timeout=60) as client:
        print("Connected!")

        # Subscribe to accelerometer characteristic
        await client.start_notify(BLE_CHARACTERISTIC_UUID, imu_handler)

        print("[INFO] Listening for data...")

        # Keep looping until spacebar is pressed
        while not stop_flag:
            await asyncio.sleep(0.1)

        print("[INFO] Stopping data collection and saving CSV.")

    # Ensure writer thread finishes
    writer_thread.join()
    print("[INFO] Data collection complete.")

if __name__ == "__main__":
    asyncio.run(main())