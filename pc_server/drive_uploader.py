import os
import pickle
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    creds = None
    token_path = 'token.pickle'
    creds_path = 'client_secrets.json'  # Descárgalo desde tu proyecto en Google Cloud Console

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # Refrescar o pedir login si no hay credenciales válidas
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

def upload_session_to_drive(session_path: str):
    session_path = Path(session_path)
    if not session_path.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {session_path}")

    drive_service = get_drive_service()

    folder_metadata = {
        'name': session_path.name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"📁 Carpeta creada en Drive: {session_path.name} (ID: {folder_id})")

    # Subir archivos individuales
    for file_path in session_path.glob("*"):
        if file_path.is_file():
            file_metadata = {
                'name': file_path.name,
                'parents': [folder_id]
            }
            media = MediaFileUpload(str(file_path), resumable=True)
            uploaded = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"✅ Subido: {file_path.name} (ID: {uploaded.get('id')})")
