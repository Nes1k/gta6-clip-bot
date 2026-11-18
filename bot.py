import os
import re
import cv2
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

# Kick yayıncıları (Türk + Global yayıncılar)
KICK_STREAMERS = [
    "rraenee",
    "elraenn",
    "kendinemuzisyen",
    "wtcn",
    "mithrain",
    "adinross",
    "xqc"
]

# Twitch'te öncelikli takip edilecek büyük yayıncılar
TWITCH_PRIORITY_STREAMERS = [
    "kaicenat",
    "ishowspeed",
    "xqc",
    "caseoh_",
    "jynxzi",
    "hasanabi",
    "lirik",
    "summit1g"
]

# Kendi yayınında yerleşik canlı altyazı kullanan yayıncılar
BUILTIN_SUB_STREAMERS = {
    "caseoh_",
    "caseoh"
}

BLACKLIST_CHANNELS = {
    "trailer_kanal_adi_1",
    "trailer_kanal_adi_2"
}

SKIP_KEYWORDS = ["trailer", "fivem", "gta 5", "gtav", "mod", "roleplay", "rp"]

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

def get_cascade(xml_name):
    xml_path = os.path.join(cv2.data.haarcascades, xml_name)
    if hasattr(cv2, "CascadeClassifier"):
        return cv2.CascadeClassifier(xml_path)
    if hasattr(cv2, "objdetect") and hasattr(cv2.objdetect, "CascadeClassifier"):
        return cv2.objdetect.CascadeClassifier(xml_path)
    return None

def has_embedded_subtitles(video_path, broadcaster=""):
    """Videonun alt orta şeridinde yayıncının kendi gömülü altyazısı var mı kontrol eder."""
    if broadcaster.lower() in BUILTIN_SUB_STREAMERS:
        return True
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or vw <= 0 or vh <= 0:
        cap.release()
        return False

    text_like_frames = 0
    for i in [2, 5, 8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * (i / 10.0)))
        ret, frame = cap.read()
        if not ret:
            continue
        roi = frame[int(vh * 0.80):int(vh * 0.95), int(vw * 0.30):int(vw * 0.70)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 300)
        edge_density = edges.mean() / 255.0
        if 0.04 < edge_density < 0.25:
            text_like_frames += 1
    cap.release()
    return text_like_frames >= 2

def detect_webcam_box(video_path, broadcaster=""):
    """Yayıncı kamerasını kafaları kesmeden tam ortalayarak bulur."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0 or vw <= 0 or vh <= 0:
        cap.release()
        return (600, 420, 1320, 260), "right"

    box_h = int(vh * 0.36)
    box_w = int(box_h * (1080 / 760))

    b_low = broadcaster.lower()
    if b_low == "rraenee":
        cap.release()
        return (box_w, box_h, vw - box_w, int(vh * 0.24)), "right"

    try:
        frontal_cascade = get_cascade("haarcascade_frontalface_default.xml")
        profile_cascade = get_cascade("haarcascade_profileface.xml")
        
        if frontal_cascade is not None:
            left_hits = []
            right_hits = []

            for i in [2, 4, 6, 8, 10]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * (i / 12.0)))
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = list(frontal_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(20, 20)))
                if not faces and profile_cascade is not None:
                    faces = list(profile_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=2, minSize=(20, 20)))
                    
                for (x, y, w, h) in faces:
                    cx, cy = x + w // 2, y + h // 2
                    if cx < vw * 0.28 and cy > vh * 0.68:
                        continue
                    if cx < vw * 0.32:
                        left_hits.append((x, y, w, h))
                    elif cx > vw * 0.68:
                        right_hits.append((x, y, w, h))

            cap.release()

            if len(right_hits) >= len(left_hits) and len(right_hits) > 0:
                avg_cx = sum(b[0] + b[2] // 2 for b in right_hits) // len(right_hits)
                avg_cy = sum(b[1] + b[3] // 2 for b in right_hits) // len(right_hits)
                crop_x = max(0, min(vw - box_w, avg_cx - box_w // 2))
                crop_y = max(0, min(vh - box_h, avg_cy - int(box_h * 0.42)))
                return (box_w, box_h, crop_x, crop_y), "right"
            elif len(left_hits) > 0:
                avg_cx = sum(b[0] + b[2] // 2 for b in left_hits) // len(left_hits)
                avg_cy = sum(b[1] + b[3] // 2 for b in left_hits) // len(left_hits)
                crop_x = max(0, min(vw - box_w, avg_cx - box_w // 2))
                crop_y = max(0, min(vh - box_h, avg_cy - int(box_h * 0.42)))
                return (box_w, box_h, crop_x, crop_y), "left"
    except Exception as e:
        print(f"Yuz tarama hatasi ({e})")
        cap.release()

    return (box_w, box_h, 0, int(vh * 0.12)), "left"

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
        no_speech_threshold=0.65,
        vad_parameters=dict(min_silence_duration_ms=450)
    )
    
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsStyle,DejaVu Sans,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,340,1

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

def get_kick_clips():
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    kick_list = []
    streamers = KICK_STREAMERS.copy()
    random.shuffle(streamers)
    
    tr_streamers = {"rraenee", "elraenn", "kendinemuzisyen", "wtcn", "mithrain"}
    
    for streamer in streamers:
        try:
            url = f"https://kick.com/api/v2/channels/{streamer}/clips?cursor=0&sort=date&time=day"
            r = scraper.get(url, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json().get("clips", [])
            for c in data[:2]:
                video_url = c.get("clip_url") or c.get("video_url")
                if not video_url:
                    continue
                kick_list.append({
                    "id": str(c["id"]),
                    "broadcaster": streamer.capitalize(),
                    "title": c.get("title", "GTA 6 Clip"),
                    "url": video_url,
                    "lang": "tr" if streamer in tr_streamers else "en",
                    "platform": "kick"
                })
        except Exception as e:
            print(f"Kick ({streamer}) cekilemedi: {e}")
    random.shuffle(kick_list)
    return kick_list

def get_twitch_clips():
    r_tok = requests.post("https://id.twitch.tv/oauth2/token", params={
        "client_id": TWITCH_CLIENT_ID,
        "client_secret": TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials"
    })
    token = r_tok.json()["access_token"]
    headers = {"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
    
    twitch_list = []
    started_at = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    r_users = requests.get("https://api.twitch.tv/helix/users", headers=headers, params={"login": TWITCH_PRIORITY_STREAMERS})
    for u in r_users.json().get("data", []):
        r_uclips = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
            "broadcaster_id": u["id"],
            "started_at": started_at,
            "first": 5
        })
        for clip in r_uclips.json().get("data", []):
            twitch_list.append({
                "id": clip["id"],
                "broadcaster": clip["broadcaster_name"],
                "title": clip["title"],
                "url": clip["url"],
                "lang": "en",
                "platform": "twitch"
            })

    r_game = requests.get("https://api.twitch.tv/helix/games", headers=headers, params={"name": "Grand Theft Auto VI"})
    data = r_game.json().get("data", [])
    if data:
        game_id = data[0]["id"]
        r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
            "game_id": game_id,
            "started_at": started_at,
            "first": 40
        })
        for clip in r.json().get("data", []):
            if clip.get("language") != "en" or clip.get("view_count", 0) < 5:
                continue
            twitch_list.append({
                "id": clip["id"],
                "broadcaster": clip["broadcaster_name"],
                "title": clip["title"],
                "url": clip["url"],
                "lang": "en",
                "platform": "twitch"
            })

    random.shuffle(twitch_list)
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
        base_title = random.choice(YT_TITLES_EN)
        title = f"{broadcaster} - {base_title}"[:95]
        desc = f"{broadcaster} Grand Theft Auto VI (GTA 6) best moments and reactions.\n\n#gta6 #gtavi #{broadcaster.lower()} #shorts"

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

def process_clip(clip, drive, youtube, existing, whisper_model):
    broadcaster = clip["broadcaster"]
    if broadcaster.lower() in BLACKLIST_CHANNELS:
        return False
        
    title_lower = clip["title"].lower()
    if any(kw in title_lower for kw in SKIP_KEYWORDS):
        return False

    clip_id = clip["id"]
    safe_title = "".join(c for c in clip["title"] if c.isalnum() or c in " -_")[:30].strip()
    filename = f"[{clip['platform'].upper()}]_{broadcaster}_{safe_title}_{clip_id[:6]}.mp4"
    
    if filename in existing:
        return False
        
    raw_file = f"raw_{clip_id}.mp4"
    ass_file = f"sub_{clip_id}.ass"
    
    try:
        subprocess.run(["yt-dlp", "-o", raw_file, clip["url"]], check=True)
        lang = clip["lang"]
        
        use_subs = not has_embedded_subtitles(raw_file, broadcaster)
        if use_subs:
            generate_shorts_subtitles(raw_file, ass_file, whisper_model, lang=lang)
            sub_filter = f",ass={ass_file}"
        else:
            print(f"{broadcaster} kendi altyazisina sahip -> Ek altyazi kapatildi")
            sub_filter = ""
        
        (cw, ch, cx, cy), cam_side = detect_webcam_box(raw_file, broadcaster)
        game_crop_x = "(iw-1080)/3" if cam_side == "right" else "2*(iw-1080)/3"
        
        vf_filter = (
            f"split[cam][game];"
            f"[cam]crop={cw}:{ch}:{cx}:{cy},scale=1080:760[top];"
            f"[game]scale=-1:1160,crop=1080:1160:{game_crop_x}:0[bottom];"
            f"[top][bottom]vstack=inputs=2{sub_filter}"
        )
        
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_file, "-t", "58",
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

        return True
    except Exception as e:
        print(f"Klip islenirken hata olustu ({broadcaster}), siradakine geciliyor: {e}")
        return False
    finally:
        for temp in (raw_file, ass_file, filename):
            if os.path.exists(temp):
                os.remove(temp)

def main():
    kick_clips = get_kick_clips()
    twitch_clips = get_twitch_clips()
    
    drive = build("drive", "v3", credentials=get_creds(GDRIVE_REFRESH_TOKEN))
    youtube = build("youtube", "v3", credentials=get_creds(YT_REFRESH_TOKEN))
    
    existing = get_existing_files(drive)
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    twitch_done = False
    for clip in twitch_clips:
        if process_clip(clip, drive, youtube, existing, whisper_model):
            twitch_done = True
            break

    kick_done = False
    for clip in kick_clips:
        if process_clip(clip, drive, youtube, existing, whisper_model):
            kick_done = True
            break

    if not twitch_done:
        for clip in kick_clips:
            if process_clip(clip, drive, youtube, existing, whisper_model):
                break
    elif not kick_done:
        for clip in twitch_clips:
            if process_clip(clip, drive, youtube, existing, whisper_model):
                break

if __name__ == "__main__":
    main()
