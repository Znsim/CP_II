import os
import csv
import time
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# =========================
# 1. 기본 설정
# =========================

DATA_DIR = "dataset"

CONSONANTS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'none']
VOWELS     = ['ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅔ', 'ㅚ', 'ㅟ', 'ㅢ', 'ㅑ', 'ㅕ', 'ㅛ', 'ㅠ', 'ㅒ', 'ㅖ']
CLASSES    = CONSONANTS + VOWELS

SAMPLE_EVERY   = 3   # N 프레임마다 1개 추출
RECORD_SECONDS = 10  # 키 한 번 누를 때 녹화 시간(초)

KEY_MAP_CONSONANT = {
    ord('1'): 'ㄱ', ord('2'): 'ㄴ', ord('3'): 'ㄷ', ord('4'): 'ㄹ',
    ord('5'): 'ㅁ', ord('6'): 'ㅂ', ord('7'): 'ㅅ', ord('8'): 'ㅇ',
    ord('9'): 'ㅈ', ord('a'): 'ㅊ', ord('b'): 'ㅋ', ord('c'): 'ㅌ',
    ord('d'): 'ㅍ', ord('e'): 'ㅎ', ord('0'): 'none',
}

KEY_MAP_VOWEL = {
    ord('1'): 'ㅏ', ord('2'): 'ㅓ', ord('3'): 'ㅗ', ord('4'): 'ㅜ',
    ord('5'): 'ㅡ', ord('6'): 'ㅣ', ord('7'): 'ㅐ', ord('8'): 'ㅔ',
    ord('9'): 'ㅚ', ord('a'): 'ㅟ', ord('b'): 'ㅢ', ord('c'): 'ㅑ',
    ord('d'): 'ㅕ', ord('e'): 'ㅛ', ord('f'): 'ㅠ', ord('g'): 'ㅒ',
    ord('h'): 'ㅖ',
}

RECT       = (80, 40, 560, 440)
FONT_PATH  = "C:/Windows/Fonts/malgun.ttf"
font_ui    = ImageFont.truetype(FONT_PATH, 26)
font_rec   = ImageFont.truetype(FONT_PATH, 28)

# =========================
# 2. 폴더 생성 & 카운터
# =========================

for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls, "landmarks_npy"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, cls, "landmarks_csv"), exist_ok=True)

counters = {}
for cls in CLASSES:
    npy_dir = os.path.join(DATA_DIR, cls, "landmarks_npy")
    counters[cls] = len([f for f in os.listdir(npy_dir) if f.endswith(".npy")])

# =========================
# 3. MediaPipe 초기화
# =========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# 4. 유틸 함수
# =========================

def put_kr(frame, text, pos, font, color):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def extract_landmarks(result):
    if not result.multi_hand_landmarks:
        return None
    coords = [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]
    return np.array(coords, dtype=np.float32)

def save_sample(label, landmarks):
    file_id  = f"{label}_{counters[label]:04d}"
    npy_path = os.path.join(DATA_DIR, label, "landmarks_npy", f"{file_id}.npy")
    csv_path = os.path.join(DATA_DIR, label, "landmarks_csv", f"{file_id}.csv")
    np.save(npy_path, landmarks)
    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "z"])
        for idx, (x, y, z) in enumerate(landmarks):
            writer.writerow([idx, x, y, z])
    counters[label] += 1
    return file_id

def draw_guide(frame, recording=False):
    x1, y1, x2, y2 = RECT
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if recording else (0, 255, 0), 2)

def draw_landmarks(frame, result):
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

def make_key_hint(key_map):
    return "  ".join(f"{chr(k)}:{v}" for k, v in key_map.items())

# =========================
# 5. 메인 루프
# =========================

cap  = cv2.VideoCapture(0)
mode = 'consonant'  # 'consonant' | 'vowel'

print("===== 데이터 수집 시작 =====")
print("m : 자음/모음 모드 전환  |  q : 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display = frame.copy()

    draw_landmarks(display, result)
    draw_guide(display, recording=False)

    key_map     = KEY_MAP_CONSONANT if mode == 'consonant' else KEY_MAP_VOWEL
    mode_label  = '[ 자음 모드 ]' if mode == 'consonant' else '[ 모음 모드 ]'
    mode_color  = (0, 255, 0) if mode == 'consonant' else (255, 180, 0)

    display = put_kr(display, f"{mode_label}  m: 모드전환  q: 종료", (10, 8), font_ui, mode_color)
    display = put_kr(display, make_key_hint(key_map), (10, display.shape[0] - 38), font_ui, (200, 200, 200))

    cv2.imshow("Korean Finger Spelling Data Collection", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('m'):
        mode = 'vowel' if mode == 'consonant' else 'consonant'
        print(f"모드 전환 → {'모음' if mode == 'vowel' else '자음'}")
        continue
    if key not in key_map:
        continue

    # ── 녹화 구간 ──────────────────────────────
    label = key_map[key]
    print(f"\n[{label}] 녹화 시작 ({RECORD_SECONDS}초)")

    start_time  = time.time()
    frame_idx   = 0
    saved_count = 0

    while time.time() - start_time < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        result  = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preview = frame.copy()

        draw_landmarks(preview, result)
        draw_guide(preview, recording=True)

        remaining = RECORD_SECONDS - (time.time() - start_time)
        preview = put_kr(preview,
                         f"REC [{label}]  {remaining:.1f}s  saved: {saved_count}",
                         (10, 8), font_rec, (0, 0, 255))

        cv2.imshow("Korean Finger Spelling Data Collection", preview)
        cv2.waitKey(1)

        if frame_idx % SAMPLE_EVERY == 0:
            landmarks = extract_landmarks(result)
            if landmarks is not None:
                file_id = save_sample(label, landmarks)
                saved_count += 1
                print(f"  saved: {file_id}")

        frame_idx += 1

    print(f"[{label}] 완료 — {saved_count}개 저장 (누적 {counters[label]}개)")

# =========================
# 6. 종료
# =========================

cap.release()
hands.close()
cv2.destroyAllWindows()
