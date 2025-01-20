import asyncio
import csv
from bleak import BleakClient, BleakScanner
import struct

# UUIDs for the service and characteristics
SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
ACCEL_CHARACTERISTIC_UUID = "00000000-0000-0000-0000-000000002c06"
GYRO_CHARACTERISTIC_UUID = "00002c09-0000-1000-8000-00805f9b34fb"
device_address = "53:8B:05:A1:16:52"

# CSV file setup
csv_filename = "data/imu_data.csv"
with open(csv_filename, mode="w", newline="") as csv_file:
    fieldnames = ['gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()  # Write the column headers

# Function to handle accelerometer data
def accel_handler(sender, data):
    print(f"Accelerometer Data (raw): {data}")  # Print raw data for debugging
    print(f"Data length: {len(data)}")  # Check length of the data

    # Assuming the data format is a string of numbers like "0.5, 0.3, 0.4"
    try:
        # If data is a string, split by commas and convert to floats
        accelx, accely, accelz = map(float, data.decode('utf-8').split(','))
        print(f"Accel: x={accelx}, y={accely}, z={accelz}")
        
        # Save data to CSV
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz'])
            writer.writerow({'gyrox': '', 'gyroy': '', 'gyroz': '', 'accelx': accelx, 'accely': accely, 'accelz': accelz})
    except Exception as e:
        print(f"Error decoding accelerometer data: {e}")

# Function to handle gyroscope data
def gyro_handler(sender, data):
    print(f"Gyroscope Data (raw): {data}")  # Print raw data for debugging
    print(f"Data length: {len(data)}")  # Check length of the data
    
    try:
        # If data is a string, split by commas and convert to floats
        gyrox, gyroy, gyroz = map(float, data.decode('utf-8').split(','))
        print(f"Gyro: x={gyrox}, y={gyroy}, z={gyroz}")
        
        # Save data to CSV
        with open(csv_filename, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['gyrox', 'gyroy', 'gyroz', 'accelx', 'accely', 'accelz'])
            writer.writerow({'gyrox': gyrox, 'gyroy': gyroy, 'gyroz': gyroz, 'accelx': '', 'accely': '', 'accelz': ''})
    except Exception as e:
        print(f"Error decoding gyroscope data: {e}")

# Main function to discover devices and start notifications
async def main():
    print("Scanning for devices...")
    devices = await BleakScanner.discover()

    for device in devices:
        print(f"Device found: {device.name} ({device.address})")
        if device.name == "IMU Sensor":  # Match the advertised name
            print(f"Connecting to {device.name} ({device.address})...")
            async with BleakClient(device.address, timeout=60) as client:
                print("Connected!")
                
                # Subscribe to accelerometer characteristic
                await client.start_notify(ACCEL_CHARACTERISTIC_UUID, accel_handler)

                # Subscribe to gyroscope characteristic
                await client.start_notify(GYRO_CHARACTERISTIC_UUID, gyro_handler)

                print("Listening for data... Press Ctrl+C to exit.")
                await asyncio.sleep(60)  # Keep connection alive for 60 seconds

if __name__ == "__main__":
    asyncio.run(main())
