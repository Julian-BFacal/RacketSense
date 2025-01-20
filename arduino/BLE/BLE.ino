#include <ArduinoBLE.h>
#include "LSM6DS3.h"
#include "Wire.h"

// Create an instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A); // I2C device address 0x6A

// Define BLE service and characteristics
BLEService imuService("12345678-1234-1234-1234-123456789abc"); // Custom UUID for IMU service

BLECharacteristic accelCharacteristic("2C06", BLERead | BLENotify, 20); // Accelerometer data
BLECharacteristic gyroCharacteristic("2C09", BLERead | BLENotify, 20); // Gyroscope data

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU
  if (myIMU.begin() != 0) {
    Serial.println("IMU initialization failed!");
    while (1);
  }
  Serial.println("IMU initialized successfully!");

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE module failed!");
    while (1);
  }

  // Set BLE local name and add service
  BLE.setLocalName("IMU Sensor");
  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(accelCharacteristic);
  imuService.addCharacteristic(gyroCharacteristic);
  BLE.addService(imuService);

  // Get the BLE device's MAC address
  String macAddress = BLE.address();
  Serial.print("BLE MAC Address: ");
  Serial.println(macAddress);

  // Start advertising
  BLE.advertise();
  Serial.println("BLE IMU Peripheral is now advertising...");
}

void loop() {
  // Listen for BLE central devices
  BLEDevice central = BLE.central();

  // If a central device connects
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    // While the central device is connected
    while (central.connected()) {
      // Read accelerometer data
      float accelX = myIMU.readFloatAccelX();
      float accelY = myIMU.readFloatAccelY();
      float accelZ = myIMU.readFloatAccelZ();

      // Read gyroscope data
      float gyroX = myIMU.readFloatGyroX();
      float gyroY = myIMU.readFloatGyroY();
      float gyroZ = myIMU.readFloatGyroZ();

      // Prepare accelerometer data as a string
      String accelData = String(accelX, 4) + "," + String(accelY, 4) + "," + String(accelZ, 4);
      accelCharacteristic.writeValue(accelData.c_str(), accelData.length());
      Serial.println("Accelerometer Data Sent: " + accelData);

      // Prepare gyroscope data as a string
      String gyroData = String(gyroX, 4) + "," + String(gyroY, 4) + "," + String(gyroZ, 4);
      gyroCharacteristic.writeValue(gyroData.c_str(), gyroData.length());
      Serial.println("Gyroscope Data Sent: " + gyroData);

      delay(500); // Send data every 500 ms
    }

    // If the central disconnects
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}
