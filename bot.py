import os
import time
import subprocess
import requests
from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TWITCH_CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
TWITCH_CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]
FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]

GDRIVE_CLIENT_ID = os.environ["GDRIVE_CLIENT_ID"]
GDRIVE_CLIENT_SECRET = os.environ["GDRIVE_CLIENT_SECRET"]
GDRIVE_REFRESH_TOKEN = os.environ["GDRIVE_REFRESH_TOKEN"]

def get_twitch_token():
    r = requests.post("https://id.twitch.tv/oauth2/token", params={
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials"
    })
    return r.json()["access_token"]

def get_game_id(headers, game_name):
    r = requests.get("https://api.twitch.tv/helix/games", headers=headers, params={"name": game_name})
    data = r.json().get("data", [])
    return data[0]["id"] if data else None

def get_drive_service():
    creds = Credentials(
        token=None,
        refresh_token=GDRIVE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GDRIVE_CLIENT_ID,
        client_secret=GDRIVE_CLIENT_SECRET
    )
    return build("drive", "v3", credentials=creds)

def get_existing_files(service):
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        fields="files(name)"
    ).execute()
    return {f["name"] for f in results.get("files", [])}

def upload_resumable(service, filename, folder_id):
    media = MediaFileUpload(filename, mimetype="video/mp4", chunksize=2*1024*1024, resumable=True)
    request = service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media
    )
    response = None
    while response is None:
        status, response = request.next_chunk()

def main():
    token = get_twitch_token()
    headers = {"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
    
    game_id = get_game_id(headers, "Grand Theft Auto VI") or get_game_id(headers, "Just Chatting")
    
    started_at = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
        "game_id": game_id,
        "started_at": started_at,
        "first": 15
    })
    clips = r.json().get("data", [])
    
    drive = get_drive_service()
    existing = get_existing_files(drive)
    uploaded = 0

    vf_filter = (
        "split[original][copy];"
        "[copy]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:10[blurred];"
        "[original]scale=1080:-1[scaled];"
        "[blurred][scaled]overlay=(W-w)/2:(H-h)/2"
    )

    for clip in clips:
        if uploaded >= 3:
            break
        clip_id = clip["id"]
        broadcaster = clip["broadcaster_name"]
        safe_title = "".join(c for c in clip["title"] if c.isalnum() or c in " -_")[:35].strip()
        filename = f"{broadcaster}_{safe_title}_{clip_id[:6]}.mp4"
        
        if filename in existing:
            continue
            
        raw_file = f"raw_{clip_id}.mp4"
        subprocess.run(["yt-dlp", "-o", raw_file, clip["url"]], check=True)
        
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_file,
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "26", "-c:a", "aac", filename
        ], check=True)
        
        for attempt in range(3):
            try:
                upload_resumable(drive, filename, FOLDER_ID)
                print(f"Yuklendi: {filename}")
                uploaded += 1
                break
            except Exception as e:
                print(f"Tekrar deneniyor ({attempt+1}/3): {e}")
                time.sleep(3)
                drive = get_drive_service()
        
        if os.path.exists(raw_file):
            os.remove(raw_file)
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    main()
