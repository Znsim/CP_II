"""한글 자모 분류 모델 학습 스크립트.
-> 수집 데이터 좌표 기반 학습 스크립트

dataset/ 아래의 손 랜드마크 데이터를 읽어 MLPClassifier를 학습하고
models/ 폴더에 gesture_model.pkl과 label_encoder.pkl을 저장한다.
"""

from pathlib import Path
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
TEAM_DIR = BASE_DIR / "dataset"
# 개인 수집 데이터: 모음 전체 (dataset/dataset/)
MY_DIR = TEAM_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CONSONANTS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'none']
VOWELS     = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# =========================
# 1. 데이터 로드
# =========================

X, y = [], []

def load_from(data_dir, cls):
    npy_dir = data_dir / cls / "landmarks_npy"
    if not npy_dir.exists():
        return 0
    files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]
    for f in files:
        landmarks = np.load(npy_dir / f)  # (21, 3)
        X.append(landmarks.flatten())                   # (63,)
        y.append(cls)
    return len(files)

print("=== 자음 로드 (팀원 데이터) ===")
for cls in CONSONANTS:
    cnt = load_from(TEAM_DIR, cls)
    print(f"  {cls}: {cnt}개")

print("\n=== 모음 로드 (팀원 + 개인 데이터) ===")
for cls in VOWELS:
    cnt_team = load_from(TEAM_DIR, cls)
    cnt_mine = load_from(MY_DIR, cls)
    print(f"  {cls}: {cnt_team + cnt_mine}개  (팀원:{cnt_team} + 개인:{cnt_mine})")

X = np.array(X)
y = np.array(y)

print(f"\n총 샘플: {len(X)}개  |  클래스: {len(set(y))}개")

# =========================
# 2. 인코딩 & 분할
# =========================

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"학습: {len(X_train)}개 / 테스트: {len(X_test)}개")

# =========================
# 3. 학습
# =========================

print("\n학습 중...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=False,
)
model.fit(X_train, y_train)

# =========================
# 4. 평가
# =========================

y_pred = model.predict(X_test)
print("\n=== 평가 결과 ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================
# 5. 저장
# =========================

joblib.dump(model, MODEL_DIR / "gesture_model.pkl")
joblib.dump(le,    MODEL_DIR / "label_encoder.pkl")
print("모델 저장 완료: models/gesture_model.pkl / models/label_encoder.pkl")
