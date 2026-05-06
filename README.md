# 🤟 청각장애인을 위한 대중교통 이용 도우미 키오스크

> **수어 인식 × 음성(STT) 통합 양방향 AI 소통 시스템**  
> 캡스톤디자인 II | 지도교수: 김상균

<br>

## 📌 프로젝트 개요

청각장애인이 대중교통 이용 중 역무원과 **수어 ↔ 음성**으로 실시간 양방향 소통할 수 있도록,  
Vision AI(수어 인식)와 STT(음성 인식)를 통합한 교통 안내 키오스크 시스템입니다.

> *"청각장애인은 수어로 질문하고, 역무원은 음성으로 답합니다.  
> 이 두 언어를 텍스트로 이어주는 양방향 실시간 소통 — 그것이 이 키오스크의 핵심입니다."*

<br>

## 🔍 배경 및 필요성

| 국내 청각장애인 수 | 수어통역 미제공으로 불편 | 교통시설 이용 불편 |
|:-:|:-:|:-:|
| **43만 명** | **41.1%** | **46.2%** |
| 보건복지부 2023 장애인실태조사 | 시설 이용 시 가장 높은 불편 원인 | 공공시설 중 불편 비율 상위권 |

**핵심 문제점:**
- 역무원과의 소통 단절 (수어통역 미제공 41.1%)
- 필담 방식의 비효율성
- 기존 수어 앱은 손에 들어야 해서 양손 수어 불가 (Hands-free 불가)
- RNN 기반 번역의 속도 문제 (실시간성 부족)

<br>

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🤟 **수어 인식 → 텍스트** | MediaPipe + 분류 모델로 실시간 수어 인식, 32개 한글 자모 지원 |
| 🎙️ **음성(STT) → 텍스트** | Whisper / Google STT로 음성 인식 |
| 🤖 **LLM 문장 교정** | GPT-4로 자연어 변환, 교통 도메인 특화 |
| 💬 **채팅형 양방향 UI** | 농인 ↔ 역무원 실시간 채팅 인터페이스 |
| 🚇 **교통 Open API 연동** | 실시간 열차 도착 정보, 경로 검색 제공 |
| ♿ **배리어프리 UI** | 픽토그램 + 핵심 단어 + 텍스트 |

<br>

## 🛠️ 기술 스택

### AI / ML
- **MediaPipe Holistic** — 손 21개 점, 포즈 33개 점 좌표 추출
- **scikit-learn MLPClassifier** — 32개 한글 자모 분류
- **TensorFlow / TFLite** — 경량화 모델 배포
- **Whisper / Google Cloud STT** — 음성 인식

### Backend
- **Python 3.11+**
- **FastAPI** — REST API 서버
- **SQLAlchemy** — ORM
- **SQLite** — 대화 로그 DB

### Frontend (별도 레포)
- **React 18** — UI 프레임워크
- **TypeScript** — 타입 안정성
- **Socket.IO** — 실시간 통신
- **Tailwind CSS** — 스타일링

### Infrastructure
- **Docker & Docker Compose** — 멀티컨테이너 환경
- **Nginx** — 리버스 프록시

<br>

## 🎯 성능 목표

| 항목 | 목표값 | 측정 방법 |
|------|--------|-----------|
| 수어 인식 정확도 (32개 자모) | ≥ 85% | 테스트셋 Accuracy |
| STT 정확도 (지하철 소음 환경) | ≥ 85% | WER (Word Error Rate) |
| 수어 → 텍스트 End-to-End 지연 | ≤ 2초 | 타임스탬프 로깅 |
| 음성 → 텍스트 지연 | ≤ 1.5초 | 타임스탬프 로깅 |

<br>

## 📁 프로젝트 구조

```
.
├── gesture_demo.py                    # MediaPipe 제스처 인식 데모
├── realtime_spelling_demo.py          # 실시간 수어 추론 및 단어 조합 데모
├── train_gesture_classifier.py        # 분류 모델 학습 스크립트
├── word_composer/                     # 한글 자모 조합 엔진
│   ├── korean_composer.py             #   자모 → 음절 조합 로직
│   ├── word_builder.py                #   실시간 단어 조합기 (Dwell 타이머)
│   ├── __init__.py
│   └── README.md
├── models/                            # 학습된 모델 & 리소스
│   ├── gesture_model.pkl              # 자모 분류 모델 (MLPClassifier)
│   ├── label_encoder.pkl              # 인덱스 ↔ 한글 매핑
│   └── gesture_recognizer.task        # MediaPipe 제스처 인식 모델
├── dataset/                           # 손 랜드마크 학습 데이터
│   ├── {자모}/
│   │   ├── landmarks_csv/             # CSV 형식 좌표
│   │   └── landmarks_npy/             # Numpy 형식 좌표
│   └── images/                        # 원본 영상
├── 데이터/                             # 개인 수집 데이터
│   ├── humandata/
│   │   ├── dataset.py                 # 데이터 로드 유틸
│   │   └── dataset/                   # 수집된 랜드마크
│   └── 종류/                          # 자모별 예시 이미지
├── 데이터수집_공유/                   # 팀 공동 데이터 수집 도구
│   ├── dataset.py                     # 데이터 수집 스크립트
│   ├── 수집_가이드.md
│   └── 참고이미지/
├── docs/                              # 개발 문서
│   ├── 백엔드_개발_가이드.md           # FastAPI 모델 & API 명세
│   ├── 프론트엔드_개발_가이드.md       # React 프론트엔드 설계
│   └── model_card_*.pdf               # 모델 카드
├── .gitignore
├── requirements.txt
└── README.md                          # 이 파일
```

<br>

## 🚀 빠른 시작

### 사전 요구사항

```bash
Python 3.11+
pip (또는 conda)
Webcam
```

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/Znsim/CP_II.git
cd CP_II

# 2. 가상환경 생성
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. 의존 패키지 설치
pip install -r requirements.txt
```

### 실행

#### 1️⃣ 실시간 수어 추론 데모 (권장)

웹캠 영상에서 손 랜드마크를 추출하여 자모를 인식하고, 실시간으로 한글 단어를 조합합니다.

```bash
python realtime_spelling_demo.py
```

**조작:**
- **q**: 종료
- **b**: 마지막 자모 제거
- **c**: 입력 초기화

**출력:**
- 상단: 현재 인식된 자모 + 신뢰도
- 하단: 조합 중인 음절 + 확정된 단어 + Dwell 진행도

#### 2️⃣ 분류 모델 학습

`dataset/` 폴더의 랜드마크 데이터로 새 모델을 학습하고 `models/` 에 저장합니다.

```bash
python train_gesture_classifier.py
```

**출력:**
- 클래스별 샘플 수
- 학습/테스트 분할
- 정확도 리포트
- `models/gesture_model.pkl`, `models/label_encoder.pkl` 저장

#### 3️⃣ MediaPipe 제스처 인식 데모

별도 모델(`gesture_recognizer.task`)를 사용해 동작 인식을 테스트합니다.

```bash
python gesture_demo.py
```

<br>

## 📚 주요 모듈 설명

### word_composer (한글 음절 조합 엔진)

자모 입력을 두벌식 한글 오토마타로 음절로 조합합니다.

**사용 예시:**
```python
from word_composer.korean_composer import KoreanComposer

c = KoreanComposer()
for jamo in ['ㅅ', 'ㅏ', 'ㄱ', 'ㅗ', 'ㅏ']:
    c.add(jamo)
print(c.text)  # → '사과'
```

자세한 문서는 [word_composer/README.md](word_composer/README.md) 참고

### realtime_spelling_demo.py

매 프레임 손 랜드마크를 분류 모델로 추론하고, `WordBuilder`로 Dwell 타이머를 기반으로 단어를 조합합니다.

**핵심 흐름:**
1. MediaPipe로 손 21개 점 추출
2. 점들을 flatten → 분류 모델 입력 (shape: 1×63)
3. 모델 출력: 클래스 인덱스 + 신뢰도
4. label_encoder로 인덱스 → 한글 자모 변환
5. WordBuilder에 전달 → Dwell 기반 조합 → 화면 표시

<br>

## 🔒 보안 및 개인정보 보호

- 카메라 영상 — 수어 인식 처리 후 **즉시 폐기** (디스크 저장 금지)
- 음성 데이터 — STT 처리 후 **즉시 폐기** (디스크 저장 금지)
- 대화 로그 — 세션 종료 후 **30분 이내 자동 삭제**
- 사용자 개인정보 수집 없음 (비회원 / 비로그인)
- 모든 API 통신 HTTPS (TLS 1.2+) 암호화

<br>

## ⚖️ 관련 법률 및 표준

- **장애인차별금지법** — 공공시설에서의 정당한 편의 제공 의무
- **한국 수어법** — 한국수어의 공용어 지정 및 수어 통역 서비스 확대
- **교통약자 이동편의 증진법** — 교통시설에서의 장애인 접근성 보장
- **KWCAG 2.2** — 한국형 웹 콘텐츠 접근성 지침 준수

<br>

## 📖 개발 문서

- [백엔드 개발 가이드](docs/백엔드_개발_가이드.md) — FastAPI 모델 로드, API 명세, DB 스키마
- [프론트엔드 개발 가이드](docs/프론트엔드_개발_가이드.md) — React 흐름, MediaPipe.js 연동
- [word_composer 가이드](word_composer/README.md) — 한글 자모 조합 알고리즘

<br>

## 🌱 향후 확장 방안

- 병원 접수 / 공공기관 민원 / 공항 관광 안내 등 타 도메인 확장 (B2G 상용화)
- 텍스트 → 수어 아바타 역번역 기능
- 클라우드(AWS/GCP) 배포 및 다중 키오스크 운영
- 다국어 수어 지원 (ASL 등)

<br>

## 📝 라이선스

MIT License

<br>

---

캡스톤디자인 II | 2026
