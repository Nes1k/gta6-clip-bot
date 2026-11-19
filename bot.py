import os
import re
import cv2
import time
import random
import urllib.request
import subprocess
import requests
import cloudscraper
import numpy as np
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

KICK_STREAMERS = [
    "adinross",
    "xqc",
    "n3on",
    "trainwreckstv",
    "destiny",
    "brucedropemoff"
]

TWITCH_PRIORITY_STREAMERS = [
    "kaicenat",
    "ishowspeed",
    "xqc",
    "caseoh_",
    "jynxzi",
    "plaqueboymax",
    "agent00",
    "stableronaldo",
    "lacy",
    "hasanabi",
    "lirik",
    "summit1g"
]

BUILTIN_SUB_STREAMERS = {
    "caseoh_",
    "caseoh"
}

BLACKLIST_CHANNELS = {
    "trailer_kanal_adi_1",
    "trailer_kanal_adi_2"
}

SKIP_KEYWORDS = ["trailer", "fivem", "gta 5", "gtav", "mod", "roleplay", "rp", "reroll"]

YT_TITLES_EN = [
    "GTA 6 Physics Are Actually INSANE! 🤯 #gta6 #shorts",
    "Rockstar Really Thought of Everything in GTA 6... 🔥 #gta6 #shorts",
    "This GTA 6 Detail Blew Everyone's Mind! 😱 #gtavi #shorts",
    "GTA 6 Gameplay Looks Next-Gen! 🚨 #gta6 #rockstargames #shorts",
    "No Way This Just Happened in GTA 6... 😭 #gta6 #gtavi #shorts"
]

BAD_ROOTS = [
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy",
    "cunt", "nigg", "faggot", "retard", "whore", "slut", "motherf"
]

YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

def ensure_yunet_model():
    if not os.path.exists(YUNET_MODEL_PATH):
        urllib.request.urlretrieve(YUNET_URL, YUNET_MODEL_PATH)

def has_embedded_subtitles(video_path, broadcaster=""):
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

def find_webcam_frame_bounds(frames, face_cx, face_cy, face_w, face_h, vw, vh):
    if len(frames) < 2:
        return None

    min_w = int(face_w * 2.2)
    max_w = int(vw * 0.42)
    min_h = int(face_h * 2.0)
    max_h = int(vh * 0.45)

    ref_gray = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY)
    grad_x = np.abs(cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3))

    y_band1 = max(0, face_cy - face_h)
    y_band2 = min(vh, face_cy + face_h)
    col_scores = np.mean(grad_x[y_band1:y_band2, :], axis=0)

    x_band1 = max(0, face_cx - face_w)
    x_band2 = min(vw, face_cx + face_w)
    row_scores = np.mean(grad_y[:, x_band1:x_band2], axis=1)

    if face_cx < vw * 0.22:
        left = 0
        right_search_start = max(face_cx + int(face_w * 0.8), min_w)
        right_search_end = min(vw, max_w)
        if right_search_end > right_search_start:
            right = right_search_start + int(np.argmax(col_scores[right_search_start:right_search_end]))
        else:
            right = min(vw, int(face_w * 3.5))
    elif face_cx > vw * 0.78:
        right = vw
        left_search_end = min(face_cx - int(face_w * 0.8), vw - min_w)
        left_search_start = max(0, vw - max_w)
        if left_search_end > left_search_start:
            left = left_search_start + int(np.argmax(col_scores[left_search_start:left_search_end]))
        else:
            left = max(0, vw - int(face_w * 3.5))
    else:
        return None

    top_search_start = max(0, face_cy - int(max_h * 0.65))
    top_search_end = max(top_search_start + 1, face_cy - int(face_h * 0.65))
    if face_cy < vh * 0.20:
        top = 0
    else:
        top = top_search_start + int(np.argmax(row_scores[top_search_start:top_search_end]))

    bot_search_start = min(vh - 1, face_cy + int(face_h * 0.8))
    bot_search_end = min(vh, top + max_h)
    if face_cy > vh * 0.80:
        bottom = vh
    elif bot_search_end > bot_search_start:
        bottom = bot_search_start + int(np.argmax(row_scores[bot_search_start:bot_search_end]))
    else:
        bottom = min(vh, top + int(face_h * 2.8))

    cw = right - left
    ch = bottom - top
    if cw < min_w or ch < min_h:
        return None

    return (cw, ch, left, top)

def detect_webcam_box(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0 or vw <= 0 or vh <= 0:
        cap.release()
        return None, "fullscreen"

    try:
        ensure_yunet_model()
        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (vw, vh),
            score_threshold=0.65, nms_threshold=0.3, top_k=5
        )

        sampled_frames = []
        left_faces = []
        right_faces = []
        center_big_hits = 0

        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * (i / 9.0)))
            ret, frame = cap.read()
            if not ret:
                continue
            sampled_frames.append(frame)
            _, faces = detector.detect(frame)
            if faces is None:
                continue

            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                cx, cy = x + w // 2, y + h // 2

                if vw * 0.34 < cx < vw * 0.66 and w > vw * 0.11:
                    center_big_hits += 1
                    continue

                if cx < vw * 0.36:
                    left_faces.append((x, y, w, h))
                elif cx > vw * 0.64:
                    right_faces.append((x, y, w, h))

        cap.release()

        if center_big_hits >= 3 and len(left_faces) < 2 and len(right_faces) < 2:
            return None, "fullscreen"

        hits = right_faces if len(right_faces) >= len(left_faces) else left_faces
        side = "right" if len(right_faces) >= len(left_faces) else "left"

        if len(hits) >= 2:
            avg_cx = sum(b[0] + b[2] // 2 for b in hits) // len(hits)
            avg_cy = sum(b[1] + b[3] // 2 for b in hits) // len(hits)
            avg_w = sum(b[2] for b in hits) // len(hits)
            avg_h = sum(b[3] for b in hits) // len(hits)

            exact_box = find_webcam_frame_bounds(sampled_frames, avg_cx, avg_cy, avg_w, avg_h, vw, vh)
            if exact_box is not None:
                cw, ch, cx, cy = exact_box
                return (cw, ch, cx, cy), side

            box_h = max(int(vh * 0.20), min(int(vh * 0.38), int(avg_h * 2.9)))
            box_w = min(vw, int(box_h * (1080 / 740)))
            crop_x = 0 if side == "left" else max(0, vw - box_w)
            crop_y = max(0, min(vh - box_h, avg_cy - int(box_h * 0.38)))
            return (box_w, box_h, crop_x, crop_y), side

    except Exception as e:
        print(f"YuNet tarama hatasi ({e})")
        cap.release()

    return None, "fullscreen"

def censor_word(word_str):
    stripped = re.sub(r'[^\w]', '', word_str).lower()
    if any(root in stripped for root in BAD_ROOTS):
        return word_str[0] + "***"
    return word_str

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def extract_clean_voice_wav(video_path, wav_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-t", "58",
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", "highpass=f=110,lowpass=f=4800,dynaudnorm=f=150:g=15",
        wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)

def generate_shorts_subtitles(video_path, ass_path, model):
    clean_wav = video_path + "_clean.wav"
    try:
        extract_clean_voice_wav(video_path, clean_wav)
        audio_input = clean_wav
    except Exception:
        audio_input = video_path

    segments, _ = model.transcribe(
        audio_input,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        no_speech_threshold=0.60,
        condition_on_previous_text=False,
        vad_parameters=dict(min_silence_duration_ms=350)
    )
    
    if os.path.exists(clean_wav):
        os.remove(clean_wav)

    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ShortsStyle,Montserrat,82,&H00FFFFFF,&H0000FFFF,&H00000000,&H99000000,-1,0,0,0,100,100,1,0,1,8,3,2,50,50,240,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for segment in segments:
        if not segment.words:
            continue
        valid_words = [w for w in segment.words if w.probability >= 0.42]
        
        for idx in range(0, len(valid_words), 2):
            pair = valid_words[idx:idx + 2]
            if not pair:
                continue

            w1_text = censor_word(pair[0].word.strip()).upper()
            w2_text = censor_word(pair[1].word.strip()).upper() if len(pair) > 1 else ""

            if len(pair) == 1:
                t_start = format_ass_time(pair[0].start)
                t_end = format_ass_time(pair[0].end)
                pop = r"{\fscx116\fscy116\t(0,75,\fscx100\fscy100)\c&H0000FFFF&}"
                events.append(f"Dialogue: 0,{t_start},{t_end},ShortsStyle,,0,0,0,,{pop}{w1_text}")
            else:
                t1_start = format_ass_time(pair[0].start)
                t1_end = format_ass_time(pair[1].start)
                if pair[1].start <= pair[0].start:
                    t1_end = format_ass_time((pair[0].start + pair[1].end) / 2.0)

                pop_in = r"{\fscx115\fscy115\t(0,75,\fscx100\fscy100)}"
                line1 = f"{pop_in}{{\\c&H0000FFFF&}}{w1_text} {{\\c&H00FFFFFF&}}{w2_text}"
                events.append(f"Dialogue: 0,{t1_start},{t1_end},ShortsStyle,,0,0,0,,{line1}")

                t2_start = t1_end
                t2_end = format_ass_time(pair[1].end)
                line2 = f"{{\\c&H00FFFFFF&}}{w1_text} {{\\c&H0000FFFF&}}{w2_text}"
                events.append(f"Dialogue: 0,{t2_start},{t2_end},ShortsStyle,,0,0,0,,{line2}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(events))

def get_kick_clips():
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
    kick_list = []
    streamers = KICK_STREAMERS.copy()
    random.shuffle(streamers)
    
    for streamer in streamers:
        try:
            url = f"https://kick.com/api/v2/channels/{streamer}/clips?cursor=0&sort=views&time=day"
            r = scraper.get(url, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json().get("clips", [])
            for c in data[:3]:
                if c.get("view_count", 0) < 30:
                    continue
                video_url = c.get("clip_url") or c.get("video_url")
                if not video_url:
                    continue
                kick_list.append({
                    "id": str(c["id"]),
                    "broadcaster": streamer.capitalize(),
                    "title": c.get("title", "GTA 6 Clip"),
                    "url": video_url,
                    "platform": "kick",
                    "views": c.get("view_count", 0)
                })
        except Exception as e:
            print(f"Kick ({streamer}) cekilemedi: {e}")
    random.shuffle(kick_list)
    return kick_list

def get_twitch_clips():
    twitch_list = []
    try:
        r_tok = requests.post("https://id.twitch.tv/oauth2/token", params={
            "client_id": TWITCH_CLIENT_ID,
            "client_secret": TWITCH_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }, timeout=15)
        token = r_tok.json()["access_token"]
        headers = {"Client-ID": TWITCH_CLIENT_ID, "Authorization": f"Bearer {token}"}
        
        started_at = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")

        r_users = requests.get("https://api.twitch.tv/helix/users", headers=headers, params={"login": TWITCH_PRIORITY_STREAMERS}, timeout=15)
        for u in r_users.json().get("data", []):
            r_uclips = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
                "broadcaster_id": u["id"],
                "started_at": started_at,
                "first": 5
            }, timeout=15)
            for clip in r_uclips.json().get("data", []):
                if clip.get("view_count", 0) < 25:
                    continue
                twitch_list.append({
                    "id": clip["id"],
                    "broadcaster": clip["broadcaster_name"],
                    "title": clip["title"],
                    "url": clip["url"],
                    "platform": "twitch",
                    "views": clip.get("view_count", 0)
                })

        r_game = requests.get("https://api.twitch.tv/helix/games", headers=headers, params={"name": "Grand Theft Auto VI"}, timeout=15)
        data = r_game.json().get("data", [])
        if data:
            game_id = data[0]["id"]
            r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
                "game_id": game_id,
                "started_at": started_at,
                "first": 50
            }, timeout=15)
            for clip in r.json().get("data", []):
                if clip.get("language") != "en" or clip.get("view_count", 0) < 40:
                    continue
                twitch_list.append({
                    "id": clip["id"],
                    "broadcaster": clip["broadcaster_name"],
                    "title": clip["title"],
                    "url": clip["url"],
                    "platform": "twitch",
                    "views": clip.get("view_count", 0)
                })
    except Exception as e:
        print(f"Twitch API hatasi: {e}")

    twitch_list.sort(key=lambda x: x["views"], reverse=True)
    top_pool = twitch_list[:15]
    random.shuffle(top_pool)
    return top_pool

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

def upload_to_youtube(youtube_service, filename, broadcaster=""):
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
        subprocess.run(["yt-dlp", "-o", raw_file, clip["url"]], check=True, timeout=120)
        
        use_subs = not has_embedded_subtitles(raw_file, broadcaster)
        if use_subs:
            generate_shorts_subtitles(raw_file, ass_file, whisper_model)
            sub_filter = f",ass={ass_file}"
        else:
            print(f"{broadcaster} kendi altyazisina sahip -> Ek altyazi kapatildi")
            sub_filter = ""
        
        cam_box, _ = detect_webcam_box(raw_file)
        
        if cam_box is not None:
            cw, ch, cx, cy = cam_box
            vf_filter = (
                f"split[cam][game];"
                f"[cam]crop={cw}:{ch}:{cx}:{cy},scale=1080:740:force_original_aspect_ratio=increase,crop=1080:740[top];"
                f"[game]split[gbg][gfg];"
                f"[gbg]scale=-1:1180,crop=1080:1180:(iw-1080)/2:0,boxblur=25:5[gblurred];"
                f"[gfg]scale=1080:-2[gmain];"
                f"[gblurred][gmain]overlay=0:(1180-h)/2[bottom];"
                f"[top][bottom]vstack=inputs=2{sub_filter}"
            )
        else:
            vf_filter = (
                f"split[bg][fg];"
                f"[bg]scale=-1:1920,crop=1080:1920:(iw-1080)/2:0,boxblur=25:5[blurred];"
                f"[fg]scale=1080:-2[main];"
                f"[blurred][main]overlay=(W-w)/2:(H-h)/2{sub_filter}"
            )
        
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_file, "-t", "58",
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-c:a", "aac", filename
        ], check=True, timeout=180)
        
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
            upload_to_youtube(youtube, filename, broadcaster=broadcaster)
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
    ensure_yunet_model()
    all_clips = get_twitch_clips() + get_kick_clips()
    
    drive = build("drive", "v3", credentials=get_creds(GDRIVE_REFRESH_TOKEN))
    youtube = build("youtube", "v3", credentials=get_creds(YT_REFRESH_TOKEN))
    
    existing = get_existing_files(drive)
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    uploaded = 0
    for clip in all_clips:
        if uploaded >= 2:
            break
        if process_clip(clip, drive, youtube, existing, whisper_model):
            uploaded += 1

if __name__ == "__main__":
    main()
