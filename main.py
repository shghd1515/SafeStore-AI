"""
main.py — SafeStore AI 통합 백엔드 서버
========================================
무인매장·점포 통합 안전/환경 관리 솔루션

통합 구성:
  · IoT 환경 모니터링 (DHT22 + PMS5003 → Supabase)
  · ML 예측 (XGBoost/LightGBM/GradientBoosting) + Autoencoder 이상치
  · Gemini 2.5 Flash AI 챗봇 (/chat)
  · YOLO-Pose 이상행동 감지 (vision/) → anomaly_events 테이블
  · 외부 API: 에어코리아 · 기상청 · 생활기상지수
  · 자동 스케줄러 (환기 알람, 이벤트 분류, 주간 리포트)

실행:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, json, math, joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from contextlib import asynccontextmanager
import jwt

from jwt import PyJWTError
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler

# vision 패키지 import
# stream_proxy는 cloudflared로 영상 받으므로 비활성화
# 노트북에서 라파이 직접 연결 + YOLO 추론할 때만 다시 켜기
HAS_STREAM_PROXY = False
print("[main] 영상 프록시 비활성화 (cloudflared 사용)")

# 챗봇 라우터 임포트
from chatbot import router as chatbot_router

load_dotenv()

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

engine    = get_engine()
scheduler = BackgroundScheduler()

# ── 모델 저장소 ───────────────────────────────────────────────────────────────
# MODEL_DIR은 ModelStore 정의 후 절대경로로 설정됩니다 (아래)

class ModelStore:
    model_temp        = None
    model_humi        = None
    model_pm25        = None
    scaler            = None
    feature_names     = []
    feature_names_pm25 = []  # ← 이 줄 추가
    hourly_pattern    = {}
    metadata          = {}

store = ModelStore()

# 새 폴더 구조 기준: 이 파일과 같은 디렉토리의 ml/models
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "models")

def load_models() -> bool:
    required = ["model_temp.pkl", "model_humi.pkl", "scaler.pkl", "feature_names.pkl"]
    for f in required:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            print(f"[경고] 모델 없음: {f}")
            return False
    store.model_temp    = joblib.load(os.path.join(MODEL_DIR, "model_temp.pkl"))
    store.model_humi    = joblib.load(os.path.join(MODEL_DIR, "model_humi.pkl"))
    store.scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    store.feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    pm25_feat_path = os.path.join(MODEL_DIR, "feature_names_pm25.pkl")
    if os.path.exists(pm25_feat_path):
        store.feature_names_pm25 = joblib.load(pm25_feat_path)
        print(f"[모델] PM2.5 피처: {store.feature_names_pm25}")    
    pm25_path = os.path.join(MODEL_DIR, "model_pm25.pkl")
    if os.path.exists(pm25_path):
        store.model_pm25 = joblib.load(pm25_path)
    p = os.path.join(MODEL_DIR, "hourly_pattern.json")
    if os.path.exists(p):
        store.hourly_pattern = {
            (int(x["hour"]), int(x["is_weekend"])): x
            for x in json.load(open(p, encoding="utf-8"))
        }
    m = os.path.join(MODEL_DIR, "metadata.json")
    if os.path.exists(m):
        store.metadata = json.load(open(m, encoding="utf-8"))
    print(f"[모델] 로드 완료 ({store.metadata.get('trained_at','?')})")
    return True


# ── 환기 알람 ────────────────────────────────────────────────────────────────
def ventilation_alarm():
    now = datetime.now()
    print(f"\n{'='*50}")
    print(f"[알람] 오후 2시! 지금 창문을 열어 15분 환기하세요!")
    print(f"{'='*50}\n")
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE sensor_combined SET event = '환기시작' "
                "WHERE recorded_at = (SELECT MAX(recorded_at) FROM "
                "(SELECT recorded_at FROM sensor_combined) AS t)"
            ))
            conn.commit()
        print(f"[이벤트] 환기시작 기록: {now.strftime('%H:%M')}")
    except Exception as e:
        print(f"[이벤트 오류] {e}")

def ventilation_end():
    print(f"\n[알람] 15분 환기 완료! 창문을 닫으세요.")
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE sensor_combined SET event = '환기종료' "
                "WHERE recorded_at = (SELECT MAX(recorded_at) FROM "
                "(SELECT recorded_at FROM sensor_combined) AS t)"
            ))
            conn.commit()
        print(f"[이벤트] 환기종료 기록 완료")
    except Exception as e:
        print(f"[이벤트 오류] {e}")


# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    scheduler.add_job(ventilation_alarm, "cron", hour=14, minute=0,  id="vent_start")
    scheduler.add_job(ventilation_end,   "cron", hour=14, minute=15, id="vent_end")
    scheduler.start()
    print("[스케줄러] 오후 2시 환기 알람 등록 완료")
    yield
    scheduler.shutdown()

app = FastAPI(
    title="SafeStore AI",
    description="무인매장 통합 안전·환경 관리 솔루션 (IoT + ML + Vision + AI)",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 챗봇 라우터 등록
app.include_router(chatbot_router)

# frontend 정적 파일 서빙
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    # 이미지 등 자산 파일: /assets/hero.png 형태로 직접 접근
    if os.path.exists(ASSETS_DIR):
        app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


# ── 유틸 ──────────────────────────────────────────────────────────────────────
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

def get_recommendation(hour, is_weekend, curr_temp, curr_humi, curr_pm25=0.0):
    def _pattern():
        # 1. 저장된 hourly_pattern 사용
        if store.hourly_pattern:
            p = store.hourly_pattern.get((hour, is_weekend)) or \
                store.hourly_pattern.get((hour, 0))
            if p:
                return {"temperature": p.get("target_temp", 22.0),
                        "humidity":    p.get("target_humi", 50.0),
                        "pm25":        p.get("target_pm25", 20.0)}

        # 2. DB에서 시간대별 평균값 동적 계산
        try:
            with get_engine().connect() as conn:
                row = conn.execute(text("""
                    SELECT
                        ROUND(AVG(temperature)::numeric, 1) as avg_temp,
                        ROUND(AVG(humidity)::numeric, 1)    as avg_humi,
                        ROUND(AVG(pm25)::numeric, 1)        as avg_pm25
                    FROM sensor_combined
                    WHERE EXTRACT(HOUR FROM recorded_at) = :hour
                      AND pm25 IS NOT NULL
                      AND recorded_at >= NOW() - INTERVAL '30 days'
                """), {"hour": hour}).fetchone()

                if row and row[0]:
                    return {
                        "temperature": float(row[0]),
                        "humidity":    float(row[1]),
                        "pm25":        float(row[2]),
                    }
        except Exception as e:
            print(f"[패턴 룩업 오류] {e}")

        # 3. 시간대별 기본값 (계절 고려)
        import datetime
        month = datetime.datetime.now().month
        # 봄/가을: 20°C, 여름: 25°C, 겨울: 18°C
        if month in [3, 4, 5, 9, 10, 11]:
            base_temp = 20.0
        elif month in [6, 7, 8]:
            base_temp = 25.0
        else:
            base_temp = 18.0

        # 시간대별 온도 보정
        if 6 <= hour <= 10:
            temp_adj = -1.0   # 아침은 약간 낮게
        elif 13 <= hour <= 17:
            temp_adj = +1.5   # 오후는 약간 높게
        elif 22 <= hour or hour <= 5:
            temp_adj = -2.0   # 밤은 낮게 (수면)
        else:
            temp_adj = 0.0

        return {
            "temperature": base_temp + temp_adj,
            "humidity":    50.0,
            "pm25":        20.0,
        }

    if store.model_temp is None:
        r = _pattern()
        r["method"] = "pattern_lookup"
        r["hour"]   = hour
        r["is_weekend"] = is_weekend
        return r

    dow = 5 if is_weekend else 2
    feats = {
        "hour_sin": math.sin(2*math.pi*hour/24), "hour_cos": math.cos(2*math.pi*hour/24),
        "dow_sin":  math.sin(2*math.pi*dow/7),   "dow_cos":  math.cos(2*math.pi*dow/7),
        "is_weekend": float(is_weekend),
        "temp_ma60": curr_temp, "humi_ma60": curr_humi, "pm25_ma60": curr_pm25,
        "temp_diff": 0.0, "humi_diff": 0.0, "pm25_diff": 0.0, "has_event": 0.0,
    }
    row = pd.DataFrame([[feats.get(f, 0.0) for f in store.feature_names]],
                       columns=store.feature_names)
    X_s = store.scaler.transform(row)
    X_s = pd.DataFrame(X_s, columns=store.feature_names)

    # PM2.5 모델은 별도 피처셋 사용
    if store.model_pm25 and store.feature_names_pm25:
        row_pm25 = pd.DataFrame(
            [[feats.get(f, 0.0) for f in store.feature_names_pm25]],
            columns=store.feature_names_pm25
        )
        X_pm25 = pd.DataFrame(row_pm25.values, columns=store.feature_names_pm25)
        pm25_pred = round(float(store.model_pm25.predict(X_pm25)[0]), 2)
    else:
        pm25_pred = curr_pm25

    ml = {
        "temperature": round(float(store.model_temp.predict(X_s)[0]), 2),
        "humidity":    round(float(store.model_humi.predict(X_s)[0]), 2),
        "pm25":        pm25_pred,
    }
    pat    = _pattern()
    n      = store.metadata.get("n_samples", 0)
    wm, wp = (0.3, 0.7) if n < 5000 else (0.7, 0.3)
    return {
        "temperature": round(ml["temperature"]*wm + pat["temperature"]*wp, 2),
        "humidity":    round(ml["humidity"]   *wm + pat["humidity"]   *wp, 2),
        "pm25":        round(ml["pm25"]       *wm + pat["pm25"]       *wp, 2),
        "method":      f"blend(ml={wm}, pattern={wp})",
        "hour":        hour,
        "is_weekend":  is_weekend,
    }


# ── 스키마 ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    hour: int = 12; is_weekend: int = 0
    current_temp: float = 23.0; current_humi: float = 43.0; current_pm25: float = 25.0

class EventRequest(BaseModel):
    event_name: str

# ── 인증 헬퍼 ──────────────────────────────────────────────────────────────
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

def verify_token(authorization: str = None) -> dict:
    """JWT 토큰을 검증하고 사용자 정보 반환"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization 헤더 없음")
    
    # "Bearer xxxxx" 형식에서 토큰만 추출
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload  # {"sub": user_id, "email": "...", ...}
    except PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"토큰 검증 실패: {str(e)}")

def get_user_role(user_id: str) -> str:
    """DB에서 사용자 권한 조회"""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT role FROM user_profiles WHERE id = :id"),
                {"id": user_id}
            ).fetchone()
            return result[0] if result else "pending"
    except Exception as e:
        print(f"[권한 조회 오류] {e}")
        return "pending"

def require_admin(authorization: str):
    """관리자 권한 필수 (관리자 아니면 403)"""
    payload = verify_token(authorization)
    user_id = payload.get("sub")
    role = get_user_role(user_id)
    if role != "admin":
        raise HTTPException(status_code=403, detail="관리자 권한 필요")
    return payload

def require_user(authorization: str):
    """일반 사용자 이상 권한 필수 (pending 아니면 통과)"""
    payload = verify_token(authorization)
    user_id = payload.get("sub")
    role = get_user_role(user_id)
    if role == "pending":
        raise HTTPException(status_code=403, detail="관리자 승인 대기 중")
    if role not in ("user", "admin"):
        raise HTTPException(status_code=403, detail="권한 없음")
    return payload


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/", summary="SafeStore AI 대시보드")
def dashboard():
    path = os.path.join(FRONTEND_DIR, "dashboard.html")
    print(f"[DEBUG] dashboard path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": f"dashboard.html 경로 없음: {path}"}

@app.get("/legacy", summary="기존 IoT 대시보드 (호환용)")
def legacy_dashboard():
    """기존 index.html에 익숙한 사용자를 위한 호환 라우트"""
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "index.html 없음 (legacy 대시보드 사용 안 함)"}

@app.get("/login", summary="로그인 페이지")
def login_page():
    path = os.path.join(FRONTEND_DIR, "login.html")
    print(f"[DEBUG] login path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": f"login.html 없음: {path}"}

@app.get("/admin", summary="관리자 페이지")
def admin_page():
    path = os.path.join(FRONTEND_DIR, "admin.html")
    print(f"[DEBUG] admin path: {path}, exists: {os.path.exists(path)}")
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "admin.html 없음"}

# 외부 미세먼지 API
AIR_API_KEY = "1046200dafe9143f9410798b3638c5353c7004949298293d317032ed3e415c85"
_outdoor_cache = {"data": None, "time": 0}

def get_outdoor_air():
    """외부 미세먼지 조회 (서울 기준) - 1시간 캐시"""
    import time
    now = time.time()

    # 1시간 이내 캐시 사용
    if _outdoor_cache["data"] and now - _outdoor_cache["time"] < 3600:
        return _outdoor_cache["data"]

    try:
        url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"
        params = {
            "serviceKey": AIR_API_KEY,
            "returnType": "json",
            "numOfRows": 1,
            "pageNo": 1,
            "sidoName": "서울",
            "ver": "1.0"
        }
        r = requests.get(url, params=params, timeout=10)
        print(f"[외부 미세먼지] 상태코드: {r.status_code}")

        if r.status_code != 200 or not r.text.strip():
            return _outdoor_cache["data"]

        data = r.json()
        items = data["response"]["body"]["items"]
        if items:
            item = items[0]
            result = {
                "station": item["stationName"],
                "pm25":    item["pm25Value"],
                "pm10":    item["pm10Value"],
                "grade":   item["pm25Grade"],
                "time":    item["dataTime"]
            }
            _outdoor_cache["data"] = result
            _outdoor_cache["time"] = now
            return result
    except Exception as e:
        print(f"[외부 미세먼지 오류] {e}")
    return _outdoor_cache["data"]

@app.get("/outdoor-air", summary="외부 미세먼지")
def outdoor_air():
    return get_outdoor_air() or {"error": "외부 미세먼지 조회 실패"}

@app.get("/pattern", summary="시간대별 환경 패턴 분석")
def get_pattern():
    """최근 30일 데이터 기반 시간대별 평균값 분석"""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    EXTRACT(HOUR FROM recorded_at)::int as hour,
                    ROUND(AVG(temperature)::numeric, 1) as avg_temp,
                    ROUND(AVG(humidity)::numeric, 1)    as avg_humi,
                    ROUND(AVG(pm25)::numeric, 1)        as avg_pm25,
                    ROUND(MAX(pm25)::numeric, 1)        as max_pm25,
                    ROUND(MIN(pm25)::numeric, 1)        as min_pm25,
                    COUNT(*)                            as count
                FROM sensor_combined
                WHERE pm25 IS NOT NULL
                  AND recorded_at >= NOW() - INTERVAL '30 days'
                GROUP BY hour
                ORDER BY hour ASC
            """)).fetchall()
        return [
            {
                "hour": r[0],
                "avg_temp": r[1],
                "avg_humi": r[2],
                "avg_pm25": r[3],
                "max_pm25": r[4],
                "min_pm25": r[5],
                "count": r[6]
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[패턴 오류] {e}")
        return []

@app.get("/alerts", summary="알림 히스토리")
def get_alerts():
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT alert_type, message, value, threshold, created_at
                FROM alert_logs
                ORDER BY created_at DESC
                LIMIT 50
            """)).fetchall()
        return [
            {
                "alert_type": r[0],
                "message":    r[1],
                "value":      r[2],
                "threshold":  r[3],
                "created_at": str(r[4])
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[알림 히스토리 오류] {e}")
        return []

@app.get("/correlation", summary="온습도 상관관계 분석")
def get_correlation():
    """온도-습도-PM2.5 상관관계 분석"""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    ROUND(temperature::numeric, 1) as temp,
                    ROUND(humidity::numeric, 1)    as humi,
                    ROUND(pm25::numeric, 1)        as pm25
                FROM sensor_combined
                WHERE pm25 IS NOT NULL
                  AND temperature IS NOT NULL
                  AND humidity IS NOT NULL
                  AND recorded_at >= NOW() - INTERVAL '30 days'
                ORDER BY recorded_at DESC
                LIMIT 1000
            """)).fetchall()

        data = [{"temp": float(r[0]), "humi": float(r[1]), "pm25": float(r[2])} for r in rows]

        if len(data) < 10:
            return {"error": "데이터 부족"}

        temps  = [d["temp"]  for d in data]
        humis  = [d["humi"]  for d in data]
        pm25s  = [d["pm25"]  for d in data]

        def correlation(x, y):
            n  = len(x)
            mx = sum(x) / n
            my = sum(y) / n
            num   = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            denom = (sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y)) ** 0.5
            return round(num / denom, 3) if denom else 0

        temp_humi_corr = correlation(temps, humis)
        temp_pm25_corr = correlation(temps, pm25s)
        humi_pm25_corr = correlation(humis, pm25s)

        def interpret(corr):
            if corr > 0.7:   return "강한 양의 상관관계"
            elif corr > 0.3: return "약한 양의 상관관계"
            elif corr < -0.7:return "강한 음의 상관관계"
            elif corr < -0.3:return "약한 음의 상관관계"
            else:             return "거의 상관없음"

        return {
            "temp_humi": {
                "value": temp_humi_corr,
                "interpret": interpret(temp_humi_corr),
                "desc": "온도 ↑ → 습도 ↑" if temp_humi_corr > 0 else "온도 ↑ → 습도 ↓"
            },
            "temp_pm25": {
                "value": temp_pm25_corr,
                "interpret": interpret(temp_pm25_corr),
                "desc": "온도 ↑ → PM2.5 ↑" if temp_pm25_corr > 0 else "온도 ↑ → PM2.5 ↓"
            },
            "humi_pm25": {
                "value": humi_pm25_corr,
                "interpret": interpret(humi_pm25_corr),
                "desc": "습도 ↑ → PM2.5 ↑" if humi_pm25_corr > 0 else "습도 ↑ → PM2.5 ↓"
            },
            "sample_count": len(data)
        }
    except Exception as e:
        print(f"[상관관계 오류] {e}")
        return {"error": str(e)}

@app.get("/forecast", summary="미래 환경 예측")
def get_forecast(hours: int = 3):
    """이동평균 기반 시계열 예측 (가볍고 빠름)"""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text("""
                SELECT recorded_at, temperature, humidity, pm25
                FROM sensor_combined
                WHERE pm25 IS NOT NULL
                  AND recorded_at >= NOW() - INTERVAL '24 hours'
                ORDER BY recorded_at DESC
                LIMIT 60
            """)).fetchall()

        if len(rows) < 10:
            return {"error": "데이터 부족"}

        # 최근 데이터로 이동평균 계산
        temps  = [float(r[1]) for r in rows if r[1]]
        humis  = [float(r[2]) for r in rows if r[2]]
        pm25s  = [float(r[3]) for r in rows if r[3]]

        # 최근 10개 평균 추세
        n = min(10, len(temps))
        avg_temp  = sum(temps[:n])  / n
        avg_humi  = sum(humis[:n])  / n
        avg_pm25  = sum(pm25s[:n])  / n

        # 추세 계산 (최근 10개 vs 이전 10개)
        if len(temps) >= 20:
            prev_temp = sum(temps[n:n*2]) / n
            prev_humi = sum(humis[n:n*2]) / n
            prev_pm25 = sum(pm25s[n:n*2]) / n
            trend_temp = (avg_temp - prev_temp) / n
            trend_humi = (avg_humi - prev_humi) / n
            trend_pm25 = (avg_pm25 - prev_pm25) / n
        else:
            trend_temp = trend_humi = trend_pm25 = 0

        from datetime import datetime, timedelta
        now = datetime.now()

        predictions = []
        for h in range(1, hours + 1):
            future_time = now + timedelta(hours=h)
            pred_temp = round(avg_temp + trend_temp * h, 1)
            pred_humi = round(avg_humi + trend_humi * h, 1)
            pred_pm25 = round(max(0, avg_pm25 + trend_pm25 * h), 1)

            # 건강 점수도 예측
            health = calc_health_score(pred_temp, pred_humi, pred_pm25)

            predictions.append({
                "time":        future_time.strftime("%Y-%m-%d %H:%M"),
                "hour":        f"{h}시간 후",
                "temperature": pred_temp,
                "humidity":    pred_humi,
                "pm25":        pred_pm25,
                "health_score": health["score"],
                "health_grade": health["grade"],
                "health_color": health["color"],
            })

        return {
            "forecast_hours": hours,
            "current": {
                "temperature": round(avg_temp, 1),
                "humidity":    round(avg_humi, 1),
                "pm25":        round(avg_pm25, 1),
            },
            "predictions": predictions
        }

    except Exception as e:
        print(f"[예측 오류] {e}")
        return {"error": str(e)}

# Autoencoder 이상치 감지
_ae_model  = None
_ae_scaler = None
_ae_meta   = None

def load_autoencoder():
    global _ae_model, _ae_scaler, _ae_meta
    try:
        ae_path = os.path.join(MODEL_DIR, "autoencoder.pkl")
        if os.path.exists(ae_path):
            _ae_model  = joblib.load(ae_path)
            _ae_scaler = joblib.load(os.path.join(MODEL_DIR, "autoencoder_scaler.pkl"))
            _ae_meta   = joblib.load(os.path.join(MODEL_DIR, "autoencoder_meta.pkl"))
            print(f"[Autoencoder] 로드 완료 (임계값: {_ae_meta['threshold']:.4f})")
    except Exception as e:
        print(f"[Autoencoder] 로드 실패: {e}")

load_autoencoder()

@app.get("/anomaly", summary="Autoencoder 이상치 감지")
def detect_anomaly_ae():
    try:
        if _ae_model is None:
            return {"error": "Autoencoder 모델 없음"}

        s = get_latest_sensor()
        if not s:
            return {"error": "센서 데이터 없음"}

        features = _ae_meta["features"]
        vals = []
        for f in features:
            if f == "hour_sin":
                from datetime import datetime
                vals.append(np.sin(2 * np.pi * datetime.now().hour / 24))
            elif f == "hour_cos":
                from datetime import datetime
                vals.append(np.cos(2 * np.pi * datetime.now().hour / 24))
            elif f == "temp_diff":
                vals.append(0.0)
            elif f == "humi_diff":
                vals.append(0.0)
            elif f == "pm25_diff":
                vals.append(0.0)
            elif f == "temp_ma60":
                vals.append(s.get("temperature", 22.0))
            elif f == "humi_ma60":
                vals.append(s.get("humidity", 50.0))
            elif f == "pm25_ma60":
                vals.append(s.get("pm25", 25.0))
            else:
                vals.append(s.get(f, 0.0))

        X = np.array([vals])
        X_scaled = _ae_scaler.transform(X)
        X_pred   = _ae_model.predict(X_scaled)
        error    = float(np.mean((X_scaled - X_pred) ** 2))
        threshold = _ae_meta["threshold"]
        is_anomaly = error > threshold

        return {
            "is_anomaly":  is_anomaly,
            "error":       round(error, 6),
            "threshold":   round(threshold, 6),
            "score":       round(error / threshold * 100, 1),
            "current": {
                "temperature": s.get("temperature"),
                "humidity":    s.get("humidity"),
                "pm25":        s.get("pm25"),
            }
        }
    except Exception as e:
        print(f"[Autoencoder 오류] {e}")
        return {"error": str(e)}

# 미세먼지 예보 캐시 (1시간)
_air_forecast_cache = {"data": None, "time": 0}

@app.get("/air-forecast", summary="미세먼지 예보")
def get_air_forecast():
    import time
    now = time.time()

    if _air_forecast_cache["data"] and now - _air_forecast_cache["time"] < 3600:
        return _air_forecast_cache["data"]

    try:
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMinuDustFrcstDspth"
        params = {
            "serviceKey": AIR_API_KEY,
            "returnType": "json",
            "numOfRows": 10,
            "pageNo": 1,
            "searchDate": today,
            "InformCode": "PM25",
        }
        r = requests.get(url, params=params, timeout=10)
        items = r.json()["response"]["body"]["items"]

        if items:
            item = items[0]
            result = {
                "date":         item.get("informData"),
                "grade":        item.get("informGrade", ""),
                "cause":        item.get("informCause", ""),
                "overview":     item.get("informOverall", ""),
            }
            _air_forecast_cache["data"] = result
            _air_forecast_cache["time"] = now
            return result
    except Exception as e:
        print(f"[미세먼지 예보 오류] {e}")
    return {"error": "미세먼지 예보 조회 실패"}

# 날씨 예보 캐시 (3시간)
_weather_cache = {"data": None, "time": 0}

@app.get("/weather-forecast", summary="기상청 날씨 예보")
def get_weather_forecast():
    import time, xml.etree.ElementTree as ET
    now_ts = time.time()

    if _weather_cache["data"] and now_ts - _weather_cache["time"] < 10800:
        return _weather_cache["data"]

    try:
        from datetime import datetime
        now = datetime.now()
        base_date = now.strftime("%Y%m%d")
        # 발표시간 (02, 05, 08, 11, 14, 17, 20, 23)
        hour = now.hour
        base_times = [2, 5, 8, 11, 14, 17, 20, 23]
        base_time = max([t for t in base_times if t <= hour], default=23)
        base_time_str = f"{base_time:02d}00"

        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        params = {
            "serviceKey": AIR_API_KEY,
            "returnType": "XML",
            "numOfRows": 500,
            "pageNo": 1,
            "base_date": base_date,
            "base_time": base_time_str,
            "nx": 60,
            "ny": 127,
        }
        r = requests.get(url, params=params, timeout=10)
        root = ET.fromstring(r.text)

        categories = {
            "TMP": "temperature",
            "REH": "humidity",
            "PTY": "rain_type",
            "SKY": "sky",
            "WSD": "wind_speed",
            "POP": "rain_prob"
        }

        sky_map  = {"1": "맑음", "3": "구름많음", "4": "흐림"}
        rain_map = {"0": "없음", "1": "비", "2": "비/눈", "3": "눈", "4": "소나기"}

        results = {}
        for item in root.findall(".//item"):
            cat  = item.findtext("category")
            time_str = item.findtext("fcstTime")
            val  = item.findtext("fcstValue")
            if cat in categories:
                if time_str not in results:
                    results[time_str] = {"time": time_str}
                key = categories[cat]
                if cat == "SKY":
                    results[time_str][key] = sky_map.get(val, val)
                elif cat == "PTY":
                    results[time_str][key] = rain_map.get(val, val)
                else:
                    results[time_str][key] = val

        # 현재 시간 이후 6시간치만
        now_time = now.strftime("%H00")
        forecasts = [v for k, v in sorted(results.items()) if k >= now_time][:12]

        result = {
            "base_date": base_date,
            "base_time": base_time_str,
            "forecasts": forecasts
        }
        _weather_cache["data"] = result
        _weather_cache["time"] = now_ts
        return result

    except Exception as e:
        print(f"[날씨 예보 오류] {e}")
        return {"error": str(e)}

# 생활기상지수 캐시 (3시간)
_living_cache = {"data": None, "time": 0}

@app.get("/living-index", summary="생활기상지수 (자외선+대기정체)")
def get_living_index():
    import time, xml.etree.ElementTree as ET
    now_ts = time.time()

    if _living_cache["data"] and now_ts - _living_cache["time"] < 10800:
        return _living_cache["data"]

    try:
        from datetime import datetime
        now = datetime.now()
        time_str = f"{now.strftime('%Y%m%d')}{(now.hour // 3) * 3:02d}"

        def fetch_index(endpoint):
            url = f"https://apis.data.go.kr/1360000/LivingWthrIdxServiceV4/{endpoint}"
            params = {
                "serviceKey": AIR_API_KEY,
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "XML",
                "areaNo": "1100000000",
                "time": time_str,
            }
            r = requests.get(url, params=params, timeout=10)
            root = ET.fromstring(r.text)
            item = root.find(".//item")
            return item

        # 자외선 지수
        uv_item = fetch_index("getUVIdxV4")
        uv_now  = uv_item.findtext("h0") if uv_item is not None else None
        uv_grade = "낮음"
        if uv_now:
            uv_val = int(uv_now)
            uv_grade = "낮음" if uv_val <= 2 else "보통" if uv_val <= 5 else "높음" if uv_val <= 7 else "매우높음" if uv_val <= 10 else "위험"

        # 대기정체지수
        air_item = fetch_index("getAirDiffusionIdxV4")
        air_now  = air_item.findtext("h3") if air_item is not None else None
        air_grade = "좋음"
        if air_now:
            air_val = int(air_now)
            air_grade = "좋음" if air_val <= 25 else "보통" if air_val <= 50 else "나쁨" if air_val <= 75 else "매우나쁨"

        result = {
            "uv_index":    uv_now,
            "uv_grade":    uv_grade,
            "air_stagnation": air_now,
            "air_grade":   air_grade,
            "time":        time_str,
        }
        _living_cache["data"] = result
        _living_cache["time"] = now_ts
        return result

    except Exception as e:
        print(f"[생활기상지수 오류] {e}")
        return {"error": str(e)}

@app.get("/history", summary="기간별 데이터 조회")
def get_history(range: str = "24h"):
    """range: 24h, 7d, 30d"""
    try:
        interval_map = {
            "24h": "24 hours",
            "7d":  "7 days",
            "30d": "30 days"
        }
        interval = interval_map.get(range, "24 hours")
        trunc_map = {
            "24h": "hour",
            "7d":  "hour",
            "30d": "day"
        }
        trunc = trunc_map.get(range, "hour")

        with get_engine().connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    TO_CHAR(DATE_TRUNC('{trunc}', recorded_at), 'YYYY-MM-DD HH24:00') as hour_label,
                    ROUND(AVG(temperature)::numeric, 1) as avg_temp,
                    ROUND(AVG(humidity)::numeric, 1)    as avg_humi,
                    ROUND(AVG(pm25)::numeric, 1)        as avg_pm25
                FROM sensor_combined
                WHERE recorded_at >= NOW() - INTERVAL '{interval}'
                GROUP BY hour_label
                ORDER BY hour_label ASC
            """)).fetchall()
        return [{"hour": r[0], "temperature": r[1], "humidity": r[2], "pm25": r[3]} for r in rows]
    except Exception as e:
        print(f"[history 오류] {e}")
        return []

def calc_health_score(temp, humi, pm25):
    """실내 환경 건강 위험도 점수 계산 (0~100)"""
    score = 100

    # 온도 점수 (최적: 20~24°C)
    if temp is None:
        temp_score = 50
    elif 20 <= temp <= 24:
        temp_score = 100
    elif 18 <= temp < 20 or 24 < temp <= 26:
        temp_score = 80
    elif 15 <= temp < 18 or 26 < temp <= 28:
        temp_score = 60
    elif 10 <= temp < 15 or 28 < temp <= 32:
        temp_score = 30
    else:
        temp_score = 0

    # 습도 점수 (최적: 40~60%)
    if humi is None:
        humi_score = 50
    elif 40 <= humi <= 60:
        humi_score = 100
    elif 35 <= humi < 40 or 60 < humi <= 65:
        humi_score = 80
    elif 30 <= humi < 35 or 65 < humi <= 70:
        humi_score = 60
    elif 20 <= humi < 30 or 70 < humi <= 80:
        humi_score = 30
    else:
        humi_score = 0

    # PM2.5 점수 (좋음: 0~15)
    if pm25 is None:
        pm25_score = 50
    elif pm25 <= 15:
        pm25_score = 100
    elif pm25 <= 25:
        pm25_score = 80
    elif pm25 <= 35:
        pm25_score = 60
    elif pm25 <= 50:
        pm25_score = 40
    elif pm25 <= 75:
        pm25_score = 20
    else:
        pm25_score = 0

    # 가중치: PM2.5 50%, 온도 25%, 습도 25%
    total = round(pm25_score * 0.5 + temp_score * 0.25 + humi_score * 0.25)

    if total >= 90:
        grade, color, msg = "매우 좋음", "#16A34A", "최적의 실내 환경입니다!"
    elif total >= 70:
        grade, color, msg = "좋음", "#65A30D", "쾌적한 환경입니다."
    elif total >= 50:
        grade, color, msg = "보통", "#D97706", "약간의 환경 개선이 필요합니다."
    elif total >= 30:
        grade, color, msg = "나쁨", "#DC2626", "환경 개선이 필요합니다."
    else:
        grade, color, msg = "매우 나쁨", "#7C3AED", "즉시 환기 및 조치가 필요합니다!"

    return {
        "score":      total,
        "grade":      grade,
        "color":      color,
        "message":    msg,
        "temp_score": temp_score,
        "humi_score": humi_score,
        "pm25_score": pm25_score,
    }

@app.get("/status", summary="현재 센서값 + AI 권장값", tags=["ML 제어"])
def get_status():
    s = get_latest_sensor(); now = datetime.now(); w = int(now.weekday() >= 5)
    rec = get_recommendation(now.hour, w,
                             s.get("temperature", 23.0),
                             s.get("humidity", 43.0),
                             s.get("pm25") or 25.0)
    health = calc_health_score(s.get("temperature"), s.get("humidity"), s.get("pm25"))
    return {
        "current": s,
        "recommendation": {"hour": now.hour, "is_weekend": w, **rec},
        "diff": {
            "temp_diff": round(rec["temperature"] - (s.get("temperature") or 0), 2),
            "humi_diff": round(rec["humidity"]    - (s.get("humidity") or 0),    2),
            "pm25_diff": round(rec["pm25"]        - (s.get("pm25") or 0),        2),
        },
        "health_score": health
    }

@app.post("/predict", summary="조건 기반 ML 예측", tags=["ML 제어"])
def predict(req: PredictRequest):
    return {"input": req.model_dump(),
            "output": get_recommendation(req.hour, req.is_weekend,
                                         req.current_temp, req.current_humi,
                                         req.current_pm25)}

@app.get("/schedule", summary="오늘 24시간 스케줄", tags=["ML 제어"])
def get_schedule():
    now = datetime.now(); w = int(now.weekday() >= 5); s = get_latest_sensor()
    return {
        "date": now.strftime("%Y-%m-%d"), "is_weekend": w,
        "ventilation_time": "14:00",
        "schedule": [
            {**get_recommendation(h, w, s.get("temperature",23.0),
                                  s.get("humidity",43.0), s.get("pm25") or 25.0),
             "hour": h, "is_active": h == now.hour, "is_ventilation": h == 14}
            for h in range(24)
        ]
    }

@app.post("/event", summary="이벤트 수동 기록", tags=["이벤트"])
def record_event(req: EventRequest):
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE sensor_combined SET event = :e "
                "WHERE recorded_at = (SELECT MAX(recorded_at) FROM "
                "(SELECT recorded_at FROM sensor_combined) AS t)"
            ), {"e": req.event_name})
            conn.commit()
        return {"message": f"이벤트 기록: {req.event_name}",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/retrain", summary="모델 재학습", tags=["모델"])
def retrain(background_tasks: BackgroundTasks):
    def _job():
        import subprocess, sys
        ml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
        subprocess.run([sys.executable, os.path.join(ml_dir, "preprocess.py")], check=True)
        subprocess.run([sys.executable, os.path.join(ml_dir, "train_model.py")], check=True)
        load_models()
    background_tasks.add_task(_job)
    return {"message": "재학습 시작 (백그라운드)"}

@app.get("/model/info", summary="모델 메타데이터", tags=["모델"])
def model_info():
    if not store.metadata: raise HTTPException(503, "모델 없음")
    return store.metadata


# ── 무인카페 보안 (Vision) ─────────────────────────────────────────────────────
@app.get("/anomaly-events", summary="최근 이상행동 이벤트", tags=["보안"])
def anomaly_events(limit: int = 30):
    """
    YOLO-Pose 등 vision 모델이 anomaly_events 테이블에 기록한
    이상행동(쓰러짐·장시간체류·폭행·화재·침입 등)을 최신순으로 반환.
    SafeStore 대시보드의 "실시간 이벤트" 섹션이 5초마다 호출.
    """
    try:
        sql = """SELECT id, event_type, detected_at, duration_sec, confidence,
                        snapshot_path, store_location, is_handled
                 FROM anomaly_events
                 ORDER BY detected_at DESC
                 LIMIT :lim"""
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"lim": limit}).fetchall()
        return [
            {
                "id": r[0],
                "event_type": r[1],
                "detected_at": str(r[2]) if r[2] else None,
                "duration_sec": r[3],
                "confidence": float(r[4]) if r[4] is not None else None,
                "snapshot_path": r[5],
                "store_location": r[6],
                "is_handled": r[7],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[anomaly_events 오류] {e}")
        return []

@app.get("/anomaly-events/count", summary="기간별 이벤트 개수", tags=["보안"])
def anomaly_count(hours: int = 24):
    """대시보드 KPI용: 최근 N시간 내 이벤트 개수"""
    try:
        sql = """SELECT event_type, COUNT(*)
                 FROM anomaly_events
                 WHERE detected_at >= NOW() - (:h || ' hours')::interval
                 GROUP BY event_type"""
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"h": hours}).fetchall()
        return {"hours": hours, "by_type": {r[0]: r[1] for r in rows},
                "total": sum(r[1] for r in rows)}
    except Exception as e:
        print(f"[anomaly_count 오류] {e}")
        return {"hours": hours, "by_type": {}, "total": 0}

@app.get("/video-stream", summary="실시간 CCTV 영상 (MJPEG)", tags=["보안"])
def video_stream():
    """
    라즈베리파이2호 → 노트북 프록시 → 브라우저로 실시간 영상 송출.
    multipart/x-mixed-replace 방식 (MJPEG over HTTP).
    
    사용:
        <img src="/video-stream" />
    
    브라우저가 자동으로 무한 갱신하며 영상을 표시.
    """
    if not HAS_STREAM_PROXY:
        raise HTTPException(503, "vision 모듈을 로드할 수 없습니다.")
    return StreamingResponse(
        stream_proxy.mjpeg_generator(target_fps=15),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video-status", summary="CCTV 연결 상태", tags=["보안"])
def video_status():
    """대시보드에서 연결 상태 확인용."""
    if not HAS_STREAM_PROXY:
        return {"available": False, "connected": False, "reason": "모듈 없음"}
    return {
        "available": True,
        "connected": stream_proxy.is_connected(),
        "pi_host": stream_proxy.PI_HOST,
        "pi_port": stream_proxy.PI_PORT,
    }

# ============================================================
# 사용자 인증 / 관리 API
# ============================================================

@app.get("/api/auth/me", summary="내 정보 + 권한 확인")
def get_my_info(authorization: str = Header(None)):
    """로그인한 사용자의 정보와 권한 반환"""
    payload = verify_token(authorization)
    user_id = payload.get("sub")
    email = payload.get("email")
    role = get_user_role(user_id)
    
    return {
        "id": user_id,
        "email": email,
        "role": role,
        "is_authenticated": True
    }

@app.get("/api/admin/pending-users", summary="가입 신청자 목록 (관리자 전용)")
def list_pending_users(authorization: str = Header(None)):
    """승인 대기 중인 사용자 목록"""
    require_admin(authorization)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, email, created_at, display_name "
                "FROM user_profiles "
                "WHERE role = 'pending' "
                "ORDER BY created_at DESC"
            )).fetchall()
        
        users = [
            {
                "id": str(row[0]),
                "email": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "display_name": row[3]
            }
            for row in result
        ]
        return {"count": len(users), "users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.get("/api/admin/users", summary="전체 사용자 목록 (관리자 전용)")
def list_all_users(authorization: str = Header(None)):
    """전체 사용자 목록 (권한 포함)"""
    require_admin(authorization)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, email, role, display_name, created_at, approved_at "
                "FROM user_profiles "
                "ORDER BY created_at DESC"
            )).fetchall()
        
        users = [
            {
                "id": str(row[0]),
                "email": row[1],
                "role": row[2],
                "display_name": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "approved_at": row[5].isoformat() if row[5] else None
            }
            for row in result
        ]
        return {"count": len(users), "users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/api/admin/approve-user/{user_id}", summary="사용자 승인 (관리자 전용)")
def approve_user(user_id: str, authorization: str = Header(None)):
    """가입 신청자를 일반 사용자로 승인"""
    payload = require_admin(authorization)
    admin_id = payload.get("sub")
    
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE user_profiles "
                "SET role = 'user', approved_at = NOW(), approved_by = :admin_id "
                "WHERE id = :user_id AND role = 'pending'"
            ), {"user_id": user_id, "admin_id": admin_id})
            conn.commit()
        return {"success": True, "message": "사용자 승인 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/api/admin/reject-user/{user_id}", summary="사용자 거부 (관리자 전용)")
def reject_user(user_id: str, authorization: str = Header(None)):
    """가입 신청자 삭제"""
    require_admin(authorization)
    
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM user_profiles WHERE id = :id AND role = 'pending'"
            ), {"id": user_id})
            conn.commit()
        return {"success": True, "message": "가입 거부 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/api/admin/change-role/{user_id}", summary="사용자 권한 변경 (관리자 전용)")
def change_user_role(user_id: str, request: Request, authorization: str = Header(None)):
    """사용자 권한 변경 (user ↔ admin)"""
    require_admin(authorization)
    
    body = {}
    try:
        import asyncio
        body = asyncio.run(request.json())
    except:
        pass
    
    new_role = body.get("role", "user")
    if new_role not in ("user", "admin", "pending"):
        raise HTTPException(status_code=400, detail="role은 user/admin/pending 중 하나")
    
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE user_profiles SET role = :role WHERE id = :id"
            ), {"role": new_role, "id": user_id})
            conn.commit()
        return {"success": True, "message": f"권한 변경 완료: {new_role}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")

@app.post("/api/auth/log-access", summary="접속 로그 기록")
def log_access(request: Request, authorization: str = Header(None)):
    """사용자 접속 로그 (로그인/로그아웃)"""
    try:
        payload = verify_token(authorization)
        user_id = payload.get("sub")
        email = payload.get("email")
        
        # 클라이언트가 보낸 action 받기
        import asyncio
        body = asyncio.run(request.json())
        action = body.get("action", "login")
        
        # IP 주소
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:200]
        
        with engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO user_access_logs (user_id, email, action, ip_address, user_agent) "
                "VALUES (:uid, :email, :action, :ip, :ua)"
            ), {
                "uid": user_id,
                "email": email,
                "action": action,
                "ip": client_ip,
                "ua": user_agent
            })
            conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/admin/access-logs", summary="접속 로그 조회 (관리자 전용)")
def get_access_logs(limit: int = 100, authorization: str = Header(None)):
    """접속 로그 최근 N개"""
    require_admin(authorization)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, user_id, email, action, ip_address, created_at "
                "FROM user_access_logs "
                "ORDER BY created_at DESC "
                "LIMIT :limit"
            ), {"limit": limit}).fetchall()
        
        logs = [
            {
                "id": row[0],
                "user_id": str(row[1]) if row[1] else None,
                "email": row[2],
                "action": row[3],
                "ip_address": row[4],
                "created_at": row[5].isoformat() if row[5] else None
            }
            for row in result
        ]
        return {"count": len(logs), "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 오류: {str(e)}")