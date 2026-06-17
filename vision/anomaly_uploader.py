"""
vision/anomaly_uploader.py
이상행동 이벤트 → Supabase anomaly_events 저장 + 텔레그램 알림

기존 pose_detector.py 등 vision 모듈에서 공통으로 호출.

사용 예:
    from vision.anomaly_uploader import push_event
    push_event(
        event_type="fall",
        duration_sec=6,
        confidence=0.89,
        snapshot_path="snapshots/2026-06-10_fall.jpg",
    )
"""

import os
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ── 설정 ────────────────────────────────────────────────────────────────────
SUPABASE_URL       = os.getenv("SUPABASE_URL", "https://qazlmymqkzlgqrptjjxk.supabase.co")
SUPABASE_ANON_KEY  = os.getenv("SUPABASE_ANON_KEY", "")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
STORE_LOCATION     = os.getenv("STORE_LOCATION", "무인매장1")

# 이벤트 종류별 한글 라벨
EVENT_LABELS = {
    "fall":       "쓰러짐 감지",
    "loitering":  "장시간 체류",
    "fight":      "폭행/싸움",
    "fire":       "화재 감지",
    "intrusion":  "침입 감지",
    "theft":      "절도 의심",
    "vandalism":  "기물파손",
    "drunk":      "주취 행동",
}

# 이벤트별 쿨다운 (초) — 동일 이벤트 연속 알림 방지
_last_sent = {}
DEFAULT_COOLDOWN = 30


def push_event(event_type: str,
               duration_sec: int = 0,
               confidence: float = 0.0,
               snapshot_path: str = "",
               cooldown: int = DEFAULT_COOLDOWN) -> bool:
    """
    이상 이벤트를 Supabase에 INSERT하고 텔레그램 알림 전송.

    Parameters
    ----------
    event_type : str
        "fall" | "loitering" | "fight" | "fire" | "intrusion" 등
    duration_sec : int
        이벤트 지속 시간(초)
    confidence : float
        모델 신뢰도 (0.0 ~ 1.0)
    snapshot_path : str
        캡처 이미지 경로
    cooldown : int
        직전 동일 이벤트와의 최소 간격

    Returns
    -------
    bool
        성공 여부
    """
    now = time.time()
    last = _last_sent.get(event_type, 0)
    if now - last < cooldown:
        print(f"[{event_type}] 쿨다운 중 ({int(cooldown - (now - last))}초 남음)")
        return False

    label = EVENT_LABELS.get(event_type, event_type)
    detected_at = datetime.now(timezone.utc).isoformat()

    # 1) Supabase REST API로 INSERT
    ok_db = _insert_supabase({
        "event_type": event_type,
        "detected_at": detected_at,
        "duration_sec": duration_sec,
        "confidence": confidence,
        "snapshot_path": snapshot_path,
        "store_location": STORE_LOCATION,
        "is_handled": False,
    })

    # 2) 텔레그램 알림
    msg = (
        f"🚨 [{STORE_LOCATION}] {label}\n"
        f"시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST\n"
        f"신뢰도: {confidence*100:.1f}%\n"
        f"지속: {duration_sec}초"
    )
    ok_tg = _send_telegram(msg)

    if ok_db or ok_tg:
        _last_sent[event_type] = now
    return ok_db and ok_tg


def _insert_supabase(payload: dict) -> bool:
    if not SUPABASE_ANON_KEY:
        print("[anomaly_uploader] SUPABASE_ANON_KEY 미설정 — 스킵")
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/anomaly_events",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=payload,
            timeout=5,
        )
        if r.status_code in (200, 201, 204):
            print(f"[Supabase] {payload['event_type']} 저장 OK")
            return True
        print(f"[Supabase 오류] {r.status_code} {r.text}")
    except Exception as e:
        print(f"[Supabase 예외] {e}")
    return False


def _send_telegram(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[anomaly_uploader] 텔레그램 키 미설정 — 스킵")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=5,
        )
        if r.status_code == 200:
            print("[Telegram] 전송 OK")
            return True
        print(f"[Telegram 오류] {r.status_code} {r.text}")
    except Exception as e:
        print(f"[Telegram 예외] {e}")
    return False


if __name__ == "__main__":
    # 테스트
    push_event("fall", duration_sec=6, confidence=0.89, cooldown=0)
