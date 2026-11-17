import os
import re
import time
import random
import subprocess
import requests
from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from faster_whisper import WhisperModel

TWITCH_CLIENT_ID = os.environ["TWITCH_CLIENT_ID"]
TWITCH_CLIENT_SECRET = os.environ["TWITCH_CLIENT_SECRET"]
FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]

GDRIVE_CLIENT_ID = os.environ["GDRIVE_CLIENT_ID"]
GDRIVE_CLIENT_SECRET = os.environ["GDRIVE_CLIENT_SECRET"]
GDRIVE_REFRESH_TOKEN = os.environ["GDRIVE_REFRESH_TOKEN"]
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"]

BLACKLIST_CHANNELS = {
    "trailer_kanal_adi_1",
    "trailer_kanal_adi_2"
}

HOOK_TITLES = [
    "GTA 6 GRAPHICS ARE UNREAL!",
    "ROCKSTAR THOUGHT OF EVERYTHING!",
    "THIS GTA 6 DETAIL IS INSANE!",
    "NEXT-GEN PHYSICS IN GTA VI!",
    "WAIT FOR IT... GTA 6 IS WILD!"
]

YT_TITLES = [
    "GTA 6 Physics Are Actually INSANE! 🤯 #gta6 #shorts",
    "Rockstar Really Thought of Everything in GTA 6... 🔥 #gta6 #shorts",
    "This GTA 6 Detail Blew Everyone's Mind! 😱 #gtavi #shorts",
    "GTA 6 Gameplay Looks Next-Gen! 🚨 #gta6 #rockstargames #shorts",
    "Nobody Expected This Detail in GTA 6! 🔥 #gta6 #shorts"
]

BAD_ROOTS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy",
    "cunt", "nigg", "faggot", "retard", "whore", "slut", "motherf",
    "amk", "aq", "sik", "orospu", "piç", "yarrak", "göt", "kahpe", "ibne"
]

def censor_text(text):
    words = text.split()
    cleaned = []
    for w in words:
        stripped = re.sub(r'[^\w]', '', w).lower()
        if any(root in stripped for root in BAD_ROOTS):
            cleaned.append(w[0] + "***")
        else:
            cleaned.append(w)
    return " ".join(cleaned)

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_shorts_subtitles(video_path, ass_path, model):
    segments, _ = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsStyle,DejaVu Sans,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,440,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for segment in segments:
        if not segment.words:
            continue
        chunk = []
        for word in segment.words:
            if word.probability < 0.45:
                continue
            chunk.append(word)
            if len(chunk) == 3 or word == segment.words[-1]:
                if not chunk:
                    continue
                start_t = format_ass_time(chunk[0].start)
                end_t = format_ass_time(chunk[-1].end)
                raw_phrase = " ".join(w.word.strip() for w in chunk)
                safe_phrase = censor_text(raw_phrase).upper()
                events.append(f"Dialogue: 0,{start_t},{end_t},ShortsStyle,,0,0,0,,{safe_phrase}")
                chunk = []

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events))

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

def get_creds(refresh_token):
    return Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GDRIVE_CLIENT_ID,
        client_secret=GDRIVE_CLIENT_SECRET
    )

def get_existing_files(drive_service):
    results = drive_service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed=false",
        fields="files(name)"
    ).execute()
    return {f["name"] for f in results.get("files", [])}

def upload_to_drive(drive_service, filename, folder_id):
    media = MediaFileUpload(filename, mimetype="video/mp4", chunksize=2*1024*1024, resumable=True)
    request = drive_service.files().create(
        body={"name": filename, "parents": [folder_id]},
        media_body=media
    )
    response = None
    while response is None:
        _, response = request.next_chunk()

def upload_to_youtube(youtube_service, filename):
    title = random.choice(YT_TITLES)
    body = {
        "snippet": {
            "title": title,
            "description": "Grand Theft Auto VI (GTA 6) best moments and gameplay clips.\n\n#gta6 #gtavi #rockstargames #shorts",
            "tags": ["gta 6", "gta vi", "rockstar games", "gta 6 gameplay", "shorts"],
            "categoryId": "20"
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False
        }
    }
    media = MediaFileUpload(filename, mimetype="video/mp4", chunksize=2*1024*1024, resumable=True)
    request = youtube_service.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        _, response = request.next_chunk()
    print(f"YouTube'a yuklendi (Private): {response.get('id')}")

def main():
    token = get_twitch_token()
    headers = {"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
    
    game_id = get_game_id(headers, "Grand Theft Auto VI") or get_game_id(headers, "Just Chatting")
    
    started_at = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
        "game_id": game_id,
        "started_at": started_at,
        "first": 20
    })
    clips = r.json().get("data", [])
    
    drive = build("drive", "v3", credentials=get_creds(GDRIVE_REFRESH_TOKEN))
    youtube = build("youtube", "v3", credentials=get_creds(YT_REFRESH_TOKEN))
    
    existing = get_existing_files(drive)
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    uploaded = 0

    for clip in clips:
        if uploaded >= 2:
            break
            
        broadcaster = clip["broadcaster_name"]
        if broadcaster.lower() in BLACKLIST_CHANNELS:
            continue
            
        if "trailer" in clip["title"].lower():
            continue

        clip_id = clip["id"]
        safe_title = "".join(c for c in clip["title"] if c.isalnum() or c in " -_")[:35].strip()
        filename = f"{broadcaster}_{safe_title}_{clip_id[:6]}.mp4"
        
        if filename in existing:
            continue
            
        raw_file = f"raw_{clip_id}.mp4"
        ass_file = f"sub_{clip_id}.ass"
        
        subprocess.run(["yt-dlp", "-o", raw_file, clip["url"]], check=True)
        generate_shorts_subtitles(raw_file, ass_file, whisper_model)
        
        hook_text = random.choice(HOOK_TITLES)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        vf_filter = (
            f"split[original][copy];"
            f"[copy]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,boxblur=25:12,eq=brightness=-0.15[blurred];"
            f"[original]scale=1380:-1,crop=1080:ih:(iw-1080)/2:0[scaled];"
            f"[blurred][scaled]overlay=0:(H-h)/2,"
            f"drawtext=fontfile={font_path}:text='{hook_text}':fontcolor=black:fontsize=46:"
            f"box=1:boxcolor=yellow@0.95:boxborderw=22:x=(w-text_w)/2:y=290,"
            f"drawtext=fontfile={font_path}:text='GTA VI • CLIPS':fontcolor=white@0.85:fontsize=34:"
            f"x=(w-text_w)/2:y=1560,"
            f"ass={ass_file}"
        )
        
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_file,
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-c:a", "aac", filename
        ], check=True)
        
        for attempt in range(3):
            try:
                upload_to_drive(drive, filename, FOLDER_ID)
                print(f"Drive'a yuklendi: {filename}")
                break
            except Exception as e:
                print(f"Drive tekrar deneniyor ({attempt+1}/3): {e}")
                time.sleep(3)
                drive = build("drive", "v3", credentials=get_creds(GDRIVE_REFRESH_TOKEN))
        
        try:
            upload_to_youtube(youtube, filename)
        except Exception as e:
            print(f"YouTube yükleme atlandı (Kota veya izin durumu): {e}")

        uploaded += 1
        for temp in (raw_file, ass_file, filename):
            if os.path.exists(temp):
                os.remove(temp)

if __name__ == "__main__":
    main()
