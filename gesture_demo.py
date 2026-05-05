"""MediaPipe Gesture Recognizer 데모.

웹캠 영상에 MediaPipe gesture recognizer .task 모델을 적용해
실시간 동작 인식을 테스트하는 예제 스크립트.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gesture_recognizer.task"

# 1. MediaPipe Gesture Recognizer 설정
# 'models/gesture_recognizer.task' 모델 파일이 필요합니다. 
# 구글 AI Edge 사이트에서 다운로드 가능 (https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer?hl=ko)
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. OpenCV 영상을 MediaPipe Image 형식으로 변환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # 4. 실시간 동작 인식 수행
    gesture_recognition_result = recognizer.recognize(mp_image)

    # 5. 결과 시각화 (인식된 동작 텍스트 출력)
    if gesture_recognition_result.gestures:
        gesture = gesture_recognition_result.gestures[0][0].category_name
        cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(gesture) # 터미널 출력

    cv2.imshow('Realtime Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
