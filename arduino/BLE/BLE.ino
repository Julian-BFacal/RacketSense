#include <ArduinoBLE.h>
#include "LSM6DS3.h"
#include "Wire.h"
#include <Thread.h>
#include <ThreadController.h>

unsigned long startTime = 0; // Store BLE connection start time

struct CombinedIMUData {
  int32_t ax, ay, az; // Accelerometer
  int32_t gx, gy, gz; // Gyroscope
  uint32_t timestamp;
};

LSM6DS3 myIMU(I2C_MODE, 0x6A); // I2C device address 0x6A*

// Define BLE service and characteristics
BLEService imuService("12345678-1234-1234-1234-123456789def");
BLECharacteristic imuCharacteristic("2C06", BLERead | BLENotify, sizeof(CombinedIMUData));

// Threading
ThreadController control = ThreadController();
Thread readThread = Thread();
Thread sendThread = Thread();

CombinedIMUData imuData; // Global variable to store IMU data

void readCallback() {
  imuData = {
    static_cast<int32_t>(myIMU.readFloatAccelX() * 10000),
    static_cast<int32_t>(myIMU.readFloatAccelY() * 10000),
    static_cast<int32_t>(myIMU.readFloatAccelZ() * 10000),
    static_cast<int32_t>(myIMU.readFloatGyroX() * 10000),
    static_cast<int32_t>(myIMU.readFloatGyroY() * 10000),
    static_cast<int32_t>(myIMU.readFloatGyroZ() * 10000),
    millis() - startTime
  };
}

void sendCallback() {
  imuCharacteristic.writeValue((uint8_t*)&imuData, sizeof(imuData));
}

void setup() {
  Serial.begin(115200);

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

  // Set BLE name and advertise
  BLE.setLocalName("IMU Sensor");
  BLE.setAdvertisedService(imuService);
  imuService.addCharacteristic(imuCharacteristic);
  BLE.addService(imuService);
  BLE.setConnectionInterval(6, 6);
  BLE.advertise();

  Serial.println("BLE IMU Peripheral is now advertising...");

  // Configure threads
  readThread.onRun(readCallback);
  readThread.setInterval(10); // Read every 2ms

  sendThread.onRun(sendCallback);
  sendThread.setInterval(15); // Send every 8ms

  control.add(&readThread);
  control.add(&sendThread);
}

void loop() {
  BLEDevice central = BLE.central();
  
  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    startTime = millis();

    while (central.connected()) {
      control.run();
    }

    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}
