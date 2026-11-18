import os
import re
import time
import random
import subprocess
import requests
import cloudscraper
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

# Takip edilecek Türk Kick yayıncıları (Kullanıcı adlarını küçük harfle yaz)
KICK_STREAMERS = [
    "rraenee",
    "elraenn",
    "kendinemuzisyen",
    "wtcn",
    "mithrain"
]

BLACKLIST_CHANNELS = {
    "trailer_kanal_adi_1",
    "trailer_kanal_adi_2"
}

SKIP_KEYWORDS = ["trailer", "fivem", "gta 5", "gtav", "mod", "roleplay", "rp"]

HOOK_TITLES_EN = [
    "GTA 6 GRAPHICS ARE UNREAL!",
    "ROCKSTAR THOUGHT OF EVERYTHING!",
    "THIS GTA 6 DETAIL IS INSANE!",
    "NEXT-GEN PHYSICS IN GTA VI!",
    "WAIT FOR IT... GTA 6 IS WILD!"
]

HOOK_TITLES_TR = [
    "GTA 6 GRAFİKLERİ İNANILMAZ!",
    "ROCKSTAR HER ŞEYİ DÜŞÜNMÜŞ!",
    "BU GTA 6 DETAYI ÇOK İYİ!",
    "GTA 6 FİZİKLERİ EFSANE OLMUŞ!",
    "SONUNA KADAR İZLE... GTA 6!"
]

YT_TITLES_EN = [
    "GTA 6 Physics Are Actually INSANE! 🤯 #gta6 #shorts",
    "Rockstar Really Thought of Everything in GTA 6... 🔥 #gta6 #shorts",
    "This GTA 6 Detail Blew Everyone's Mind! 😱 #gtavi #shorts",
    "GTA 6 Gameplay Looks Next-Gen! 🚨 #gta6 #rockstargames #shorts"
]

YT_TITLES_TR = [
    "GTA 6 Fizikleri Gerçekten İnanılmaz! 🤯 #gta6 #shorts",
    "Rockstar GTA 6'da Her Şeyi Düşünmüş... 🔥 #gta6 #gtavi #shorts",
    "GTA 6'daki Bu Detay Herkesi Şaşırttı! 😱 #gta6 #shorts",
    "GTA 6 Oynanış Anları Efsane Görünüyor! 🚨 #gta6 #shorts"
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

def generate_shorts_subtitles(video_path, ass_path, model, lang="en"):
    segments, _ = model.transcribe(
        video_path,
        language=lang,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400)
    )
    
    # MarginV=510 olarak güncellendi (Sıfır kırpma modunda videonun hemen altına tam oturur)
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsStyle,DejaVu Sans,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,510,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for segment in segments:
        if not segment.words:
            continue
        chunk = []
        for word in segment.words:
            if word.probability < 0.30:
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

def get_kick_clips():
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    kick_list = []
    for streamer in KICK_STREAMERS:
        try:
            url = f"https://kick.com/api/v2/channels/{streamer}/clips?cursor=0&sort=date&time=day"
            r = scraper.get(url, timeout=15)
            if r.status_code != 200:
                print(f"Kick ({streamer}) erisim durumu: {r.status_code}")
                continue
            data = r.json().get("clips", [])
            for c in data[:3]:
                video_url = c.get("clip_url") or c.get("video_url")
                if not video_url:
                    continue
                kick_list.append({
                    "id": str(c["id"]),
                    "broadcaster": streamer.capitalize(),
                    "title": c.get("title", "GTA 6 Clip"),
                    "url": video_url,
                    "lang": "tr",
                    "platform": "kick"
                })
        except Exception as e:
            print(f"Kick ({streamer}) cekilemedi: {e}")
    return kick_list

def get_twitch_clips():
    r_tok = requests.post("https://id.twitch.tv/oauth2/token", params={
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials"
    })
    token = r_tok.json()["access_token"]
    headers = {"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
    
    r_game = requests.get("https://api.twitch.tv/helix/games", headers=headers, params={"name": "Grand Theft Auto VI"})
    data = r_game.json().get("data", [])
    if not data:
        return []
    game_id = data[0]["id"]
    
    started_at = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
        "game_id": game_id,
        "started_at": started_at,
        "first": 40
    })
    twitch_list = []
    for clip in r.json().get("data", []):
        if clip.get("language") != "en":
            continue
        if clip.get("view_count", 0) < 10:
            continue
        twitch_list.append({
            "id": clip["id"],
            "broadcaster": clip["broadcaster_name"],
            "title": clip["title"],
            "url": clip["url"],
            "lang": "en",
            "platform": "twitch"
        })
    return twitch_list

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
        pageSize=1000,
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

def upload_to_youtube(youtube_service, filename, lang="en", broadcaster=""):
    if lang == "tr":
        base_title = random.choice(YT_TITLES_TR)
        title = f"{broadcaster} - {base_title}"[:95]
        desc = f"{broadcaster} GTA 6 (Grand Theft Auto VI) en iyi anlar ve tepkiler.\n\n#gta6 #gtavi #{broadcaster.lower()} #shorts"
    else:
        title = random.choice(YT_TITLES_EN)
        desc = "Grand Theft Auto VI (GTA 6) best moments and gameplay clips.\n\n#gta6 #gtavi #rockstargames #shorts"

    body = {
        "snippet": {
            "title": title,
            "description": desc,
            "tags": ["gta 6", "gta vi", "rockstar games", "gta 6 gameplay", "shorts", broadcaster.lower()],
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
    print(f"BASARILI - YouTube'a yuklendi (Private): https://youtube.com/shorts/{response.get('id')}")

def main():
    all_clips = get_kick_clips() + get_twitch_clips()
    
    drive = build("drive", "v3", credentials=get_creds(GDRIVE_REFRESH_TOKEN))
    youtube = build("youtube", "v3", credentials=get_creds(YT_REFRESH_TOKEN))
    
    existing = get_existing_files(drive)
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    uploaded = 0

    for clip in all_clips:
        if uploaded >= 4:
            break
            
        broadcaster = clip["broadcaster"]
        if broadcaster.lower() in BLACKLIST_CHANNELS:
            continue
            
        title_lower = clip["title"].lower()
        if any(kw in title_lower for kw in SKIP_KEYWORDS):
            continue

        clip_id = clip["id"]
        safe_title = "".join(c for c in clip["title"] if c.isalnum() or c in " -_")[:30].strip()
        filename = f"[{clip['platform'].upper()}]_{broadcaster}_{safe_title}_{clip_id[:6]}.mp4"
        
        if filename in existing:
            continue
            
        raw_file = f"raw_{clip_id}.mp4"
        ass_file = f"sub_{clip_id}.ass"
        
        try:
            subprocess.run(["yt-dlp", "-o", raw_file, clip["url"]], check=True)
        except Exception as e:
            print(f"Video indirilemedi ({broadcaster}): {e}")
            continue

        lang = clip["lang"]
        generate_shorts_subtitles(raw_file, ass_file, whisper_model, lang=lang)
        
        hook_text = random.choice(HOOK_TITLES_TR if lang == "tr" else HOOK_TITLES_EN)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        # Yöntem 1: Sıfır kırpma (scale=1080:-1) ile tüm ekran ve webcamler görünür
        vf_filter = (
            f"split[original][copy];"
            f"[copy]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,boxblur=25:12,eq=brightness=-0.15[blurred];"
            f"[original]scale=1080:-1[scaled];"
            f"[blurred][scaled]overlay=0:(H-h)/2,"
            f"drawtext=fontfile={font_path}:text='{hook_text}':fontcolor=black:fontsize=46:"
            f"box=1:boxcolor=yellow@0.95:boxborderw=22:x=(w-text_w)/2:y=380,"
            f"drawtext=fontfile={font_path}:text='GTA VI • CLIPS':fontcolor=white@0.85:fontsize=34:"
            f"x=(w-text_w)/2:y=1520,"
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
            upload_to_youtube(youtube, filename, lang=lang, broadcaster=broadcaster)
        except Exception as e:
            print(f"YouTube kotasi dolu veya atlandi: {e}")

        uploaded += 1
        for temp in (raw_file, ass_file, filename):
            if os.path.exists(temp):
                os.remove(temp)

if __name__ == "__main__":
    main()
