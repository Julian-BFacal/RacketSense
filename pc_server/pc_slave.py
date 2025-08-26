import asyncio
import socket
import struct
import time
import os
import datetime
import subprocess
import logging
import configparser
import threading
from bleak import BleakScanner, BleakClient
from pathlib import Path


# ========== CARGAR CONFIGURACIÓN ==========
config = configparser.ConfigParser()
config.read('config.ini')

SERVICE_UUID = config.get("BLE", "service_uuid")
BLE_CHARACTERISTIC_UUID = config.get("BLE", "characteristic_uuid")
MAX_RETRIES = config.getint("BLE", "max_retries")
RETRY_SLEEP = config.getint("BLE", "retry_sleep")

PORT = config.getint("SERVER", "port")
SOCKET_TIMEOUT = config.getint("SERVER", "socket_timeout")

LOG_FILE = config.get("LOGGING", "log_file")
LOG_LEVEL = getattr(logging, config.get("LOGGING", "log_level").upper())

# ========== INICIALIZACIÓN ==========
imu_data = []
ble_client = None
recording = True
is_paused = False
current_session_timestamp = None
script_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_dir, "real_time_script.py")

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
    
)

def get_broadcast_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        subnet = ".".join(local_ip.split(".")[:-1])
        return subnet + ".255"
    except Exception as e:
        logging.warning(f"No se pudo obtener broadcast IP, usando fallback: {e}")
        return "255.255.255.255"


def broadcast_presence():
    broadcast_ip = get_broadcast_ip()
    port = 50001
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    while True:
        try:
            message = b"PC_ANNOUNCE"
            sock.sendto(message, (broadcast_ip, port))
            logging.debug(f"📢 Enviado broadcast a {broadcast_ip}:{port}")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Error en broadcast_presence: {e}")

# ========== FUNCIONES BLE ==========
def notification_handler(sender, data):
    global is_paused
    if is_paused:
        return
    try:
        ax, ay, az, gx, gy, gz, timestamp = struct.unpack("iiiiiii", data)
        ax /= 10000.0
        ay /= 10000.0
        az /= 10000.0
        gx /= 10000.0
        gy /= 10000.0
        gz /= 10000.0
        timestamp /= 1000.0
        imu_data.append((timestamp, ax, ay, az, gx, gy, gz))
    except Exception as e:
        logging.error(f"Error decoding packet: {e}")

async def connect_and_record_with_retry(max_retries=MAX_RETRIES):
    global ble_client
    retries = 0
    while retries < max_retries:
        try:
            logging.info(f"Intentando escanear y conectar al IMU BLE (intento {retries + 1})")
            devices = await BleakScanner.discover()
            address = None
            for d in devices:
                if SERVICE_UUID.lower() in [s.lower() for s in d.metadata.get("uuids", [])]:
                    address = d.address
                    break

            if not address:
                logging.warning("IMU no encontrada. Reintentando...")
                retries += 1
                await asyncio.sleep(RETRY_SLEEP)
                continue

            ble_client = BleakClient(address)
            await ble_client.connect()
            await ble_client.start_notify(BLE_CHARACTERISTIC_UUID, notification_handler)
            logging.info("Conexión BLE establecida")
            return ble_client

        except Exception as e:
            logging.warning(f"Error al conectar: {e}")
            retries += 1
            await asyncio.sleep(RETRY_SLEEP)

    logging.error("Se superó el número máximo de reintentos al conectar BLE")
    return None

async def stop_recording():
    global recording
    recording = False
    if ble_client and ble_client.is_connected:
        try:
            await ble_client.stop_notify(BLE_CHARACTERISTIC_UUID)
            await ble_client.disconnect()
            logging.info("IMU BLE desconectado correctamente")
        except Exception as e:
            logging.error(f"Error al detener BLE: {e}")

    if current_session_timestamp and imu_data:
        Path("data").mkdir(parents=True, exist_ok=True)
        csv_data_path = Path("data") / f"imu_{current_session_timestamp}.csv"

        try:
            headers = ["timestamp", "accelx", "accely", "accelz", "gyrox", "gyroy", "gyroz"]
            with open(csv_data_path, "w", newline="") as f_data:
                writer = csv.writer(f_data)
                writer.writerow(headers)
                writer.writerows(imu_data)

            logging.info(f"✅ CSV guardado en: {csv_data_path}")
        except Exception as e:
            logging.error(f"❌ Error al guardar CSV: {e}")

        imu_data.clear()


    with open("real_time_script.log", "a") as log_file:
        subprocess.Popen(
            ["python3", script_path, current_session_timestamp],
            stdout=log_file,
            stderr=log_file
    )
def pause_recording():
    global is_paused
    is_paused = True
    logging.info("⏸️ Pausando recolección de datos IMU")

def resume_recording():
    global is_paused
    is_paused = False
    logging.info("▶️ Reanudando recolección de datos IMU")


# ========== SERVIDOR ==========
def run_server():
    HOST = ''

    while True:
        try:
            logging.info(f"Iniciando servidor BLE en puerto {PORT}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((HOST, PORT))
                server.listen(1)
                server.settimeout(None)

                while True:
                    logging.info("Esperando conexión de cliente...")
                    conn, addr = server.accept()
                    logging.info(f"Conexión aceptada desde {addr}")
                    with conn:
                        conn.settimeout(None)
                        while True:
                            try:
                                data = conn.recv(1024).decode().strip()
                                logging.info(f"📥 Comando recibido: '{data}'")
                                if not data:
                                    break

                                if data == "WHO":
                                    logging.info("Recibido comando WHO")
                                    conn.sendall(b"PC\n")
                                    conn.shutdown(socket.SHUT_RDWR)
                                    conn.close()
                                    break

                                if data == "START":
                                    global current_session_timestamp
                                    current_session_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                    logging.info(f"📂 Timestamp de sesión asignado: {current_session_timestamp}")
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    client = loop.run_until_complete(connect_and_record_with_retry())
                                    if client:
                                        conn.sendall(b"OK")
                                    else:
                                        conn.sendall(b"ERROR_BLE")

                                elif data == "video_pause":
                                    logging.info("Recibido comando video_pause")
                                    pause_recording()
                                    conn.sendall(b"OK")

                                elif data == "video_resume":
                                    logging.info("Recibido comando video_resume")
                                    resume_recording()
                                    conn.sendall(b"OK")

                                elif data == "video_stop":
                                    logging.info("Recibido comando video_stop")
                                    loop.run_until_complete(stop_recording())
                                    break
                                else:
                                    logging.warning(f"Comando desconocido: {data}")
                                    conn.sendall(b"ERROR")

                            except socket.timeout:
                                logging.warning("Timeout esperando datos del cliente")
                                break
                            except Exception as e:
                                logging.error(f"Error en comunicación con cliente: {e}")
                                break
        except Exception as e:
            logging.critical(f"Error crítico en el servidor, reiniciando en 5s: {e}")
            time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=broadcast_presence, daemon=True).start()
    run_server()
