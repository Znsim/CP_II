"""실시간 한글 자모 추론 및 단어 조합 데모.
-> 학습 데이터 좌표 기반 추론 스크립트 

웹캠 손 랜드마크를 분류 모델로 추론하고, WordBuilder로
한글 음절/단어를 실시간으로 조합해 화면에 표시한다.
"""

from pathlib import Path
import sys, os

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import joblib
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from word_composer.word_builder import WordBuilder

MODEL_DIR = BASE_DIR / "models"

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
font_large = ImageFont.truetype(FONT_PATH, 44)
font_mid   = ImageFont.truetype(FONT_PATH, 34)
font_small = ImageFont.truetype(FONT_PATH, 20)

CONF_THRESHOLD = 0.75   # 이 이상일 때만 단어 조합에 반영

CONSONANTS = {'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'}
VOWELS     = {'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅚ','ㅛ','ㅜ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'}

def put_text_kr(frame, text, pos, font, color):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_char_type(label):
    if label in CONSONANTS: return "자음"
    if label in VOWELS:     return "모음"
    return ""

# =========================
# 1. 모델 로드
# =========================

model = joblib.load(MODEL_DIR / "gesture_model.pkl")
le    = joblib.load(MODEL_DIR / "label_encoder.pkl")
print(f"인식 가능 클래스 ({len(le.classes_)}개): {list(le.classes_)}")

# =========================
# 2. MediaPipe 초기화
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

RECT = (80, 60, 560, 420)

# =========================
# 3. 추론 함수
# =========================

def extract_landmarks(result):
    if not result.multi_hand_landmarks:
        return None
    coords = [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]
    return np.array(coords, dtype=np.float32)

def predict(landmarks):
    x = landmarks.flatten().reshape(1, -1)
    pred  = model.predict(x)[0]
    proba = model.predict_proba(x)[0].max()
    label = le.inverse_transform([pred])[0]
    return label, proba

# =========================
# 4. 단어 조합기 초기화
# =========================

builder = WordBuilder(dwell=1.0, space_dwell=2.0)

# =========================
# 5. 메인 루프
# =========================

cap = cv2.VideoCapture(0)
print("실시간 추론 시작  |  b: 지우기  c: 초기화  q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    result  = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    display = frame.copy()

    # 랜드마크 그리기
    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    # 가이드 사각형
    x1, y1, x2, y2 = RECT
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 예측
    landmarks = extract_landmarks(result)
    if landmarks is not None:
        label, confidence = predict(landmarks)
        char_type  = get_char_type(label)
        type_tag   = f"[{char_type}] " if char_type else ""
        jamo_text  = f"{type_tag}{label}  ({confidence*100:.1f}%)"
        jamo_color = (0, 255, 0) if confidence >= 0.8 else (0, 165, 255)
        feed_label = label if confidence >= CONF_THRESHOLD else None
    else:
        jamo_text  = "No hand  (2초 유지 → 공백)"
        jamo_color = (150, 150, 150)
        feed_label = None

    # WordBuilder 업데이트
    wb = builder.update(feed_label)

    # 현재 자모 (카메라 상단)
    display = put_text_kr(display, jamo_text, (x1, 12), font_large, jamo_color)

    # ── 하단 단어 표시 바 ──────────────────────────────────────
    bar = np.zeros((130, w, 3), dtype=np.uint8)

    # Dwell 진행 바
    bar_fill  = int((w - 20) * wb['progress'])
    bar_color = (0, 200, 100) if feed_label else (80, 80, 80)
    cv2.rectangle(bar, (10, 6), (w - 10, 20), (40, 40, 40), -1)
    if bar_fill > 0:
        cv2.rectangle(bar, (10, 6), (10 + bar_fill, 20), bar_color, -1)

    # 조합 중인 음절
    comp_str = f"조합 중: 【{wb['composing']}】" if wb['composing'] else "조합 중: 【 】"
    bar = put_text_kr(bar, comp_str, (10, 24), font_small, (160, 160, 160))

    # 단어 (최대 12글자, 넘으면 앞 잘라냄)
    word_display = wb['text'][-12:] if len(wb['text']) > 12 else wb['text']
    word_display = word_display if word_display.strip() else "..."
    bar = put_text_kr(bar, word_display, (10, 52), font_mid, (255, 240, 80))

    # 단축키 안내
    bar = put_text_kr(bar, "b: 지우기    c: 초기화    q: 종료",
                      (10, 104), font_small, (100, 100, 100))

    # 카메라 + 단어 바 합치기
    display = np.vstack([display, bar])

    cv2.imshow("Finger Spelling", display)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('b'): builder.backspace()
    elif key == ord('c'): builder.clear()

cap.release()
hands.close()
cv2.destroyAllWindows()
