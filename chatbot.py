"""
backend/chatbot.py
OpenAI API 기반 챗봇 엔드포인트
main.py 에 통합하여 사용

추가할 엔드포인트:
  POST /chat        → 챗봇 메시지 전송
  GET  /chat/sensor → 현재 센서 상태 (frontend용)
  GET  /chat/events → 이벤트 기록 조회
  GET  /chat/logs   → 자동제어 로그 조회
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from openai import OpenAI

load_dotenv()

router = APIRouter(prefix="/chat", tags=["OpenAI 챗봇"])

# ── OpenAI 설정 ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 기본: 빠르고 저렴
openai_client  = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── DB 연결 ───────────────────────────────────────────────────────────────────
# 전역 엔진 (연결 풀 재사용)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        supabase_url = os.getenv("SUPABASE_DB_URL")
        print(f"[DB DEBUG] SUPABASE_DB_URL exists: {bool(supabase_url)}")
        if supabase_url:
            print(f"[DB DEBUG] Connecting to Supabase: {supabase_url[:50]}...")
            _engine = create_engine(
                supabase_url,
                pool_pre_ping=True,
                pool_size=2,
                max_overflow=3,
                pool_recycle=300,
            )
        else:
            print("[DB DEBUG] SUPABASE_DB_URL 비어있음 → MySQL fallback 시도")
            url = (
                f"mysql+pymysql://{os.getenv('DB_USER','root01')}:{os.getenv('DB_PASSWORD','00000')}"
                f"@{os.getenv('DB_HOST','192.168.101.2')}:{os.getenv('DB_PORT','3307')}"
                f"/{os.getenv('DB_NAME','sensor_db')}?charset=utf8mb4"
            )
            _engine = create_engine(url, pool_pre_ping=True)
    return _engine

engine = get_engine()


# ── DB 유틸 ───────────────────────────────────────────────────────────────────
def get_latest_sensor() -> dict:
    sql = """SELECT recorded_at, temperature, humidity, pm1, pm25, pm10, event
             FROM sensor_combined ORDER BY recorded_at DESC LIMIT 1"""
    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        if row:
            return {"recorded_at": str(row[0]), "temperature": row[1],
                    "humidity": row[2], "pm1": row[3],
                    "pm25": row[4], "pm10": row[5], "event": row[6]}
    except Exception as e:
        print(f"[DB 오류] {e}")
    return {}


def get_hourly_data() -> list:
    sql = """SELECT 
                TO_CHAR(DATE_TRUNC('hour', recorded_at), 'YYYY-MM-DD HH24:00') as hour_label,
                ROUND(AVG(temperature)::numeric, 1) as avg_temp,
                ROUND(AVG(humidity)::numeric, 1)    as avg_humi,
                ROUND(AVG(pm25)::numeric, 1)        as avg_pm25
             FROM sensor_combined
             WHERE recorded_at >= NOW() - INTERVAL '24 hours'
             GROUP BY hour_label
             ORDER BY hour_label ASC"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return [{"hour": str(r[0]), "temperature": r[1],
                 "humidity": r[2], "pm25": r[3]} for r in rows]
    except Exception as e:
        print(f"[DB 오류] {e}")
    return []


def get_event_list() -> list:
    sql = """SELECT recorded_at, temperature, humidity, pm25, pm10, event
             FROM sensor_combined WHERE event IS NOT NULL
             ORDER BY recorded_at DESC LIMIT 20"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return [{"recorded_at": str(r[0]), "temperature": r[1],
                 "humidity": r[2], "pm25": r[3],
                 "pm10": r[4], "event": r[5]} for r in rows]
    except Exception as e:
        print(f"[DB 오류] {e}")
    return []

def get_anomaly_events() -> list:
    """이상행동 이벤트 조회 (화재, 쓰러짐, 폭행 등) - 한국 시간"""
    sql = """SELECT event_type, 
                    TO_CHAR(detected_at AT TIME ZONE 'Asia/Seoul', 'YYYY-MM-DD HH24:MI:SS') as kst_time,
                    duration_sec, 
                    confidence
             FROM anomaly_events
             ORDER BY detected_at DESC
             LIMIT 20"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return [{"event_type": r[0], "detected_at": r[1],
                 "duration": r[2], "confidence": r[3]} for r in rows]
    except Exception as e:
        print(f"[DB 오류] {e}")
    return []


def get_today_anomaly_summary() -> dict:
    """오늘 이상행동 발생 요약"""
    sql = """SELECT event_type, COUNT(*) as cnt
             FROM anomaly_events
             WHERE detected_at >= NOW() - INTERVAL '24 hours'
             GROUP BY event_type"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception as e:
        print(f"[DB 오류] {e}")
    return {}

def get_control_logs() -> list:
    sql = """SELECT logged_at, hour_of_day, current_temp, current_humi,
                    current_pm25, target_temp, target_humidity, target_pm25, action
             FROM control_log
             ORDER BY logged_at DESC LIMIT 20"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return [{"logged_at": str(r[0]), "hour": r[1],
                 "current_temp": r[2], "current_humi": r[3],
                 "current_pm25": r[4], "target_temp": r[5],
                 "target_humi": r[6], "target_pm25": r[7],
                 "action": r[8]} for r in rows]
    except Exception as e:
        print(f"[DB 오류] {e}")
    return []


def pm25_grade(val) -> str:
    if val is None: return "알수없음"
    if val < 15:    return "좋음"
    elif val < 35:  return "보통"
    elif val < 75:  return "나쁨"
    else:           return "매우나쁨"


# ── OpenAI 컨텍스트 빌드 ─────────────────────────────────────────────────────
def build_context() -> str:
    """시스템 프롬프트용 컨텍스트 빌드"""
    s              = get_latest_sensor()
    events         = get_event_list()[:5]
    hourly         = get_hourly_data()
    anomaly_events = get_anomaly_events()[:10]  # 최근 10개 이상행동
    anomaly_today  = get_today_anomaly_summary()  # 오늘 요약
    now            = datetime.now()

    pm25_val = s.get("pm25") or 0
    temp_val = s.get("temperature") or 0
    humi_val = s.get("humidity") or 0

    event_text = "\n".join(
        f"  {e['recorded_at']}: {e['event']} (온도 {e['temperature']}°C, PM2.5 {e['pm25']})"
        for e in events
    ) or "없음"

    # 이상행동 이벤트 텍스트
    EVENT_KOREAN = {
        "fall": "쓰러짐",
        "loitering": "장시간 체류",
        "fire": "화재",
        "smoke": "연기",
        "fight": "폭행/싸움",
        "violence": "폭행",
        "intrusion": "침입",
        "theft": "절도 의심",
    }
    
    anomaly_text = "\n".join(
        f"  {a['detected_at']}: {EVENT_KOREAN.get(a['event_type'], a['event_type'])} "
        f"(신뢰도 {(a['confidence'] or 0)*100:.0f}%, 지속 {a['duration']}초)"
        for a in anomaly_events
    ) or "없음"
    
    # 오늘 이상행동 요약
    today_summary_text = "오늘 발생 없음"
    if anomaly_today:
        today_summary_text = ", ".join(
            f"{EVENT_KOREAN.get(k, k)} {v}회"
            for k, v in anomaly_today.items()
        )

    # 24시간 패턴 분석
    pattern_text = "데이터 없음"
    trend_text   = "데이터 없음"
    if len(hourly) >= 3:
        pm25_list = [h["pm25"] for h in hourly if h["pm25"] is not None]
        temp_list = [h["temperature"] for h in hourly if h["temperature"] is not None]
        if pm25_list:
            avg_pm25 = sum(pm25_list) / len(pm25_list)
            max_pm25 = max(pm25_list)
            min_pm25 = min(pm25_list)
            pattern_text = f"PM2.5 평균 {avg_pm25:.1f} / 최고 {max_pm25:.1f} / 최저 {min_pm25:.1f} μg/m³"
        if temp_list:
            avg_temp = sum(temp_list) / len(temp_list)
            pattern_text += f"\n  온도 평균 {avg_temp:.1f}°C (24시간 기준)"

        # 트렌드 분석 (최근 3시간 vs 이전 3시간)
        if len(hourly) >= 6:
            recent  = hourly[:3]
            prev    = hourly[3:6]
            r_pm25  = [h["pm25"] for h in recent if h["pm25"] is not None]
            p_pm25  = [h["pm25"] for h in prev   if h["pm25"] is not None]
            r_temp  = [h["temperature"] for h in recent if h["temperature"] is not None]
            p_temp  = [h["temperature"] for h in prev   if h["temperature"] is not None]
            if r_pm25 and p_pm25:
                pm25_trend = sum(r_pm25)/len(r_pm25) - sum(p_pm25)/len(p_pm25)
                trend_text = f"PM2.5 {'상승 ↑' if pm25_trend > 2 else '하락 ↓' if pm25_trend < -2 else '안정 →'} ({pm25_trend:+.1f} μg/m³)"
            if r_temp and p_temp:
                temp_trend = sum(r_temp)/len(r_temp) - sum(p_temp)/len(p_temp)
                trend_text += f" / 온도 {'상승 ↑' if temp_trend > 0.5 else '하락 ↓' if temp_trend < -0.5 else '안정 →'} ({temp_trend:+.1f}°C)"

    # 건강 위험도 점수 계산
    def calc_score(t, h, p):
        ts = 100 if 20<=t<=24 else 80 if 18<=t<=26 else 60 if 15<=t<=28 else 30
        hs = 100 if 40<=h<=60 else 80 if 35<=h<=65 else 60 if 30<=h<=70 else 30
        ps = 100 if p<=15 else 80 if p<=25 else 60 if p<=35 else 40 if p<=50 else 20 if p<=75 else 0
        return round(ps*0.5 + ts*0.25 + hs*0.25)

    health_score = calc_score(temp_val, humi_val, pm25_val)
    health_grade = "매우 좋음" if health_score>=90 else "좋음" if health_score>=70 else "보통" if health_score>=50 else "나쁨" if health_score>=30 else "매우 나쁨"

    # Autoencoder 이상치 점수
    ae_text = "정보 없음"
    try:
        import requests as req
        ae_res  = req.get("http://localhost:10000/anomaly", timeout=3)
        ae_data = ae_res.json()
        if ae_data.get("is_anomaly"):
            ae_text = f"⚠️ 이상 감지! 이상 점수 {ae_data['score']}점 (정상 범위 초과)"
        else:
            ae_text = f"✅ 정상 패턴 (이상 점수 {ae_data.get('score', 0)}점)"
    except:
        pass

    # 환경 개선 추천
    recommendations = []
    if pm25_val >= 35:
        recommendations.append(f"PM2.5 {pm25_val}μg/m³ - 즉시 환기 권장 (목표: 15μg/m³ 이하)")
    if pm25_val >= 75:
        recommendations.append("PM2.5 매우 나쁨 - 마스크 착용 및 외출 자제 권장")
    if temp_val >= 28:
        recommendations.append(f"온도 {temp_val}°C - 냉방 또는 환기로 24°C 이하로 낮추세요")
    if temp_val <= 15:
        recommendations.append(f"온도 {temp_val}°C - 난방으로 20°C 이상으로 높이세요")
    if humi_val >= 70:
        recommendations.append(f"습도 {humi_val}% - 제습기 사용으로 60% 이하로 낮추세요")
    if humi_val <= 30:
        recommendations.append(f"습도 {humi_val}% - 가습기 사용으로 40% 이상으로 높이세요")
    if not recommendations:
        recommendations.append("현재 실내 환경은 전반적으로 양호합니다.")

    recommendation_text = "\n".join(f"  - {r}" for r in recommendations)

    return f"""
당신은 SafeStore AI - 무인매장 통합 안전·환경 관리 시스템의 AI 어시스턴트입니다.
사용자의 실내 환경 데이터를 분석하고 친절하고 실용적으로 답변하세요.
답변은 한국어로 2~4문장으로 간결하게 작성하세요.
구체적인 수치를 언급하며 실질적인 행동을 추천하세요.

## 현재 실내 환경 ({now.strftime('%Y-%m-%d %H:%M')})
- 온도: {temp_val}°C
- 습도: {humi_val}%
- PM2.5: {pm25_val} μg/m³ ({pm25_grade(pm25_val)})
- PM10: {s.get('pm10', 'N/A')} μg/m³
- 건강 위험도: {health_score}점 ({health_grade})

## 24시간 환경 패턴
{pattern_text}

## 최근 3시간 트렌드
{trend_text}

## AI 이상치 감지 (Autoencoder)
{ae_text}

## AI 환경 개선 추천
{recommendation_text}

## 최근 환경 이벤트
{event_text}

## 🚨 보안·안전 이상행동 이벤트 (CCTV AI 감지)
### 오늘 발생 요약 (최근 24시간)
{today_summary_text}

### 최근 이상행동 이벤트 (최근 10개)
{anomaly_text}

### 감지 가능한 이상행동
- 화재(fire): YOLO-Pose + YOLOv8 화재 감지 모델 (정확도 95%)
- 쓰러짐(fall): YOLO-Pose 자세 분석 (정확도 90%)
- 장시간 체류(loitering): 동일 위치 60초 이상
- 비영업시간 침입(intrusion): 22시~06시 사람 감지
- 폭행/싸움(violence): YOLO-Pose 기반 휴리스틱 (베타 단계)

## AI 자동제어 시스템 정보
- 매일 오후 2시 15분 환기 알람 자동 실행
- 1분마다 온도·습도·미세먼지 자동 제어 판단
- XGBoost/LightGBM ML 모델로 시간대별 최적 환경값 예측
- Autoencoder로 비정상 환경 패턴 자동 감지
- 미세먼지 기준: 좋음(0~15) / 보통(15~35) / 나쁨(35~75) / 매우나쁨(75+)

## 답변 가이드
- 현재 수치와 건강 위험도 점수를 언급하며 구체적으로 답변하세요
- 트렌드가 악화 중이면 선제적 조치를 권장하세요
- AI 이상치 감지 결과가 있으면 언급하세요
- 환기 필요 시 예상 효과를 수치로 제시하세요
- 행동 추천은 명확하고 실용적으로 하세요
- 화재/쓰러짐/폭행 등 이상행동 질문 시 위 "보안·안전 이상행동 이벤트" 데이터를 활용하세요
- 시간, 종류, 빈도, 신뢰도를 구체적으로 언급하세요
""".strip()


# ── 스키마 ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []


# ── 이벤트 자동 감지 키워드 ──────────────────────────────────────────────────
EVENT_KEYWORDS = {
    "환기시작": ["환기 시작", "환기시작", "창문 열었", "창문 열어", "환기할게"],
    "환기종료": ["환기 끝", "환기종료", "창문 닫았", "창문 닫아", "환기 완료"],
    "요리시작": ["요리 시작", "요리시작", "요리 중", "밥 하"],
    "요리종료": ["요리 끝", "요리 완료", "요리종료"],
    "청소시작": ["청소 시작", "청소기", "청소시작"],
    "청소종료": ["청소 끝", "청소 완료", "청소종료"],
    "외출":     ["외출", "나가요", "나갑니다"],
    "귀가":     ["귀가", "들어왔", "돌아왔"],
}

def detect_event(message: str):
    msg = message.lower()
    for event_name, keywords in EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in msg:
                return event_name
    return None

def record_event_db(event_name: str):
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE sensor_combined SET event = :e "
                "WHERE recorded_at = (SELECT MAX(recorded_at) FROM "
                "(SELECT recorded_at FROM sensor_combined) AS t)"
            ), {"e": event_name})
            conn.commit()
    except Exception as e:
        print(f"[이벤트 오류] {e}")


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.post("", summary="챗봇 메시지 전송")
def chat(req: ChatRequest):
    # 이벤트 자동 감지
    detected_event = detect_event(req.message)
    if detected_event:
        record_event_db(detected_event)

    if not openai_client:
        return {
            "answer": "OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.",
            "detected_event": detected_event,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # OpenAI messages 형식 구성
    # 1) system 메시지 (컨텍스트)
    # 2) 이전 대화 히스토리
    # 3) 현재 사용자 메시지
    messages = [
        {"role": "system", "content": build_context()}
    ]
    
    # 이전 대화 히스토리 추가 (최근 6개만)
    if req.history:
        for h in req.history[-6:]:
            role = h.get("role", "user")
            # OpenAI는 'user' / 'assistant' 사용
            if role in ("user", "assistant"):
                messages.append({
                    "role": role,
                    "content": h.get("content", "")
                })
    
    # 현재 사용자 메시지
    messages.append({"role": "user", "content": req.message})

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        print(f"[OpenAI 오류] {err}")
        if "rate_limit" in err.lower() or "429" in err:
            answer = "현재 AI 서비스 이용량이 초과됐습니다. 잠시 후 다시 시도해주세요. (보통 1분 후 자동 복구됩니다)"
        elif "404" in err or "not found" in err.lower() or "model" in err.lower():
            answer = "AI 모델 연결에 문제가 발생했습니다. 서버 관리자에게 문의해주세요."
        elif "api" in err.lower() or "key" in err.lower() or "auth" in err.lower():
            answer = "OpenAI API 키를 확인해주세요. .env 파일의 OPENAI_API_KEY를 확인해주세요."
        elif "billing" in err.lower() or "quota" in err.lower() or "credit" in err.lower():
            answer = "OpenAI 크레딧이 부족합니다. 결제 또는 충전 후 다시 시도해주세요."
        else:
            answer = "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    return {
        "answer":         answer,
        "detected_event": detected_event,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/sensor", summary="현재 센서 상태 + 시간대별 데이터")
def get_sensor_data():
    sensor  = get_latest_sensor()
    hourly  = get_hourly_data()
    pm25_val = sensor.get("pm25") or 0
    return {
        "current":    sensor,
        "pm25_grade": pm25_grade(pm25_val),
        "hourly":     hourly,
    }


@router.get("/events", summary="이벤트 기록 조회")
def get_events():
    return get_event_list()


@router.get("/logs", summary="자동제어 로그 조회")
def get_logs():
    return get_control_logs()
