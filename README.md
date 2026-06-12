# 🛡️ SafeStore AI

> 무인매장·점포를 위한 통합 안전·환경 관리 솔루션
>
> IoT 환경 모니터링 + ML 예측 + YOLO 이상행동 감지 + Gemini AI 챗봇

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)

---

## 📌 개요

**SafeStore AI**는 무인매장에서 발생할 수 있는 환경 위험(온습도·미세먼지)과 보안 위험(이상행동·화재)을 하나의 시스템에서 통합 관리하는 솔루션입니다.

### 핵심 기능
- 🌡️ **실내 환경 모니터링**: 온도·습도·미세먼지 실시간 측정 및 ML 예측
- 🚨 **이상행동 감지**: YOLO-Pose 기반 쓰러짐·장시간 체류 감지
- 🔥 **화재 감지**: AI 영상 분석으로 화재 징후 즉시 알림
- 📱 **통합 관제**: 단일 대시보드에서 모든 매장 상황 관리
- 🤖 **AI 챗봇**: Gemini 기반 자연어 환경 질의응답

---

## 🏗️ 시스템 아키텍처

```
[라즈베리파이1호]                [라즈베리파이2호]
DHT22 + PMS5003                  웹캠 (9999 포트)
       │                                │
       ▼                                ▼
[Supabase PostgreSQL]            [노트북 GPU]
sensor_combined                  YOLO-Pose 추론
                                       │
                                       ▼
                              anomaly_events 저장
                                       │
                  ┌────────────────────┘
                  ▼
        [FastAPI Backend (Render)]
        - /status, /history, /forecast
        - /outdoor-air, /weather-forecast
        - /anomaly-events
        - /video-stream (MJPEG)
        - /chat (Gemini)
                  │
                  ▼
        [Dashboard + Telegram Bot]
```

---

## 🛠️ 기술 스택

- **하드웨어**: Raspberry Pi 5 (×2), DHT22, PMS5003, 웹캠
- **백엔드**: FastAPI, Python 3.11
- **DB**: Supabase PostgreSQL
- **ML**: XGBoost, LightGBM, GradientBoosting, scikit-learn
- **Vision**: YOLOv8-Pose, OpenCV
- **AI**: Google Gemini 2.5 Flash
- **프론트엔드**: Vanilla JS, Chart.js
- **배포**: Render (Web Service)
- **알림**: Telegram Bot API

---

## 📁 프로젝트 구조

```
safestore_backend/
├── main.py                  # FastAPI 메인 서버
├── chatbot.py               # Gemini AI 챗봇 라우터
├── auto_scheduler.py        # 환기 알람, 이벤트 분류 스케줄러
├── ml/                      # 머신러닝
│   ├── preprocess.py
│   ├── train_model.py
│   ├── autoencoder.py
│   └── models/              # 학습된 .pkl 파일들
├── vision/                  # 영상 처리
│   ├── stream_proxy.py      # 라파이 영상 프록시
│   ├── pose_detector.py     # YOLO-Pose 이상행동
│   └── anomaly_uploader.py  # 이벤트 → Supabase + Telegram
├── frontend/
│   ├── dashboard.html       # 메인 통합 대시보드
│   └── assets/              # 이미지 자산
├── start_server.bat         # Windows 더블클릭 실행
├── requirements.txt
├── Procfile                 # Render 배포 설정
├── runtime.txt              # Python 버전 명시
└── .env.example             # 환경변수 템플릿
```

---

## ⚙️ 설치 및 실행

### 로컬 개발

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일 열어서 실제 값 입력

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

또는 Windows에서 `start_server.bat` 더블클릭.

### Render 배포

1. GitHub에 push
2. Render 대시보드에서 새 Web Service 생성
3. 환경 변수 설정 (Settings → Environment)
4. 자동 배포

---

## 🗺️ 주요 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 대시보드 UI |
| GET | `/status` | 현재 센서값 + AI 권장값 + 건강점수 |
| GET | `/forecast?hours=3` | 환경 예측 (ML) |
| GET | `/anomaly` | 환경 이상치 (Autoencoder) |
| GET | `/anomaly-events` | 이상행동 이벤트 (Vision) |
| GET | `/outdoor-air` | 에어코리아 실외 미세먼지 |
| GET | `/weather-forecast` | 기상청 날씨 |
| GET | `/history?range=24h` | 시계열 데이터 |
| GET | `/video-stream` | 실시간 CCTV (MJPEG) |
| GET | `/video-status` | CCTV 연결 상태 |
| POST | `/chat` | Gemini AI 챗봇 |

---

## 📊 머신러닝 모델

| 항목 | 모델 | CV MAE |
|------|------|--------|
| 온도 예측 | GradientBoosting | 0.0717 |
| 습도 예측 | GradientBoosting | 0.1163 |
| PM2.5 예측 | LightGBM | 9.9132 |
| 환경 이상치 | Autoencoder | 임계값 0.101 |

### 피처 엔지니어링
시간 주기성 (sin/cos), 이동 통계 (ma120, std10), 변화율 (diff5, diff10), 가속도 (accel), 교차 피처 (temp×humi) 등 12개 → 39개 피처 확장.

---

## 🚨 이상행동 감지

YOLO-Pose 기반 행동 인식:

- **쓰러짐 (Fall)**: 어깨-엉덩이 몸통선 각도 < 30°가 5초 이상 지속
- **장시간 체류 (Loitering)**: 박스 중심이 40px 내에 1800초 이상

향후 추가 예정:
- 폭행 / 싸움 / 절도 / 기물파손 / 강도 / 데이트폭력 / 납치 / 주취행동 / 침입 / 투기 / 실신 / 배회
- 화재 감지

---

## 👤 개발자

노홍욱 | [GitHub](https://github.com/shghd1515)

---

## 📝 라이선스

MIT
