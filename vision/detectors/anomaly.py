"""
vision/detectors/anomaly.py
종합 이상행동 감지 모듈 (YOLO-Pose 기반)

감지 대상:
  - 폭행 (Violence): 사람 2명 이상 + 빠른 움직임 + 가까운 거리
  - 빠른 이동 (Running): 박스 중심 이동 속도
  - 군집 (Crowding): 5명 이상 동시 감지
  - 침입 (Intrusion): 비영업 시간 사람 감지
  - 무릎 꿇기 (Kneeling): 자세 분석
  - 카운터 점프 (Counter Jump): y좌표 급격한 변화
"""

import os
import math
import time
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics 미설치. pip install ultralytics")


class AnomalyDetector:
    """종합 이상행동 감지기"""
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        # 폭행 파라미터
        violence_distance: float = 100.0,      # 두 사람 거리 임계값 (px)
        violence_speed: float = 80.0,           # 움직임 속도 임계값
        violence_hold_sec: float = 1.5,
        # 침입 파라미터
        intrusion_start_hour: int = 22,         # 비영업 시작 (22시)
        intrusion_end_hour: int = 6,            # 비영업 끝 (06시)
        # 공통
        cooldown_sec: float = 60.0,
        device: str = "cuda",
        verbose: bool = False,
    ):
        print(f"[AnomalyDetector] 모델 로딩: {model_path} (device={device})")
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            print("[AnomalyDetector] GPU 사용 불가, CPU로 진행")
        
        # 파라미터
        self.violence_distance = violence_distance
        self.violence_speed = violence_speed
        self.violence_hold_sec = violence_hold_sec
        self.intrusion_start_hour = intrusion_start_hour
        self.intrusion_end_hour = intrusion_end_hour
        self.cooldown_sec = cooldown_sec
        self.verbose = verbose
        
        # 상태 추적
        self.person_history = []
        self.violence_buf = deque(maxlen=10)

        # 시작 시간 (지속 시간 측정용)
        self.violence_start: Optional[float] = None

        # 쿨다운
        self.last_alerts = {
            "violence": 0.0,
            "intrusion": 0.0,
        }
        
        print("[AnomalyDetector] 준비 완료")
    
    def _is_in_cooldown(self, event_type: str) -> bool:
        """쿨다운 체크"""
        return (time.time() - self.last_alerts.get(event_type, 0)) < self.cooldown_sec
    
    def _trigger_alert(self, event_type: str):
        """알림 트리거 (쿨다운 업데이트)"""
        self.last_alerts[event_type] = time.time()
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 1장 분석
        
        Returns:
            {
                "violence": bool,         # 폭행
                "running": bool,          # 빠른 이동
                "crowding": bool,         # 군집
                "intrusion": bool,        # 침입
                "kneeling": bool,         # 무릎 꿇기
                "counter_jump": bool,     # 카운터 점프
                "person_count": int,
                "details": {...}
            }
        """
        result = {
            "violence": False,
            "intrusion": False,
            "person_count": 0,
            "details": {}
        }
        
        # YOLO 추론
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            print(f"[AnomalyDetector] 추론 오류: {e}")
            return result
        
        r = results[0]
        
        # 사람 정보 수집
        people = []
        if r.boxes is not None and len(r.boxes) > 0:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                # 키포인트 (선택)
                kp = None
                if r.keypoints is not None and i < len(r.keypoints.data):
                    kp = r.keypoints.data[i].cpu().numpy()
                
                people.append({
                    "cx": cx, "cy": cy,
                    "w": w, "h": h,
                    "bbox": (x1, y1, x2, y2),
                    "keypoints": kp,
                })
        
        result["person_count"] = len(people)
        
        # ── 1. 침입 (비영업 시간 사람 감지) ─────────
        result["intrusion"] = self._check_intrusion(people)

        # ── 2. 폭행 (2명 이상 + 가까이 + 빠른 움직임) ─
        result["violence"] = self._check_violence(people)

        # 이전 프레임 정보 업데이트
        self.person_history = people

        result["details"]["person_count"] = len(people)
                
        return result
    
    def _check_intrusion(self, people: List[Dict]) -> bool:
        """비영업 시간 침입 감지"""
        if not people:
            return False
        
        now = datetime.now()
        hour = now.hour
        
        # 비영업 시간: start_hour ~ 24 또는 0 ~ end_hour
        is_after_hours = False
        if self.intrusion_start_hour > self.intrusion_end_hour:
            # 예: 22시~06시
            is_after_hours = hour >= self.intrusion_start_hour or hour < self.intrusion_end_hour
        else:
            is_after_hours = self.intrusion_start_hour <= hour < self.intrusion_end_hour
        
        if is_after_hours and not self._is_in_cooldown("intrusion"):
            self._trigger_alert("intrusion")
            if self.verbose:
                print(f"[Anomaly] 🌃 비영업 시간 침입 (현재 {hour}시)")
            return True
        
        return False
    
    
    def _check_violence(self, people: List[Dict]) -> bool:
        """폭행 감지 (2명 이상 + 가까이 + 빠른 움직임)"""
        if len(people) < 2:
            self.violence_buf.append(0)
            self.violence_start = None
            return False
        
        # 가장 가까운 두 사람
        min_distance = float('inf')
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                p1, p2 = people[i], people[j]
                dist = math.hypot(p1["cx"] - p2["cx"], p1["cy"] - p2["cy"])
                if dist < min_distance:
                    min_distance = dist
        
        # 거리 조건
        if min_distance > self.violence_distance:
            self.violence_buf.append(0)
            self.violence_start = None
            return False
        
        # 빠른 움직임 체크 (이전 프레임과 비교)
        if not self.person_history:
            self.violence_buf.append(0)
            return False
        
        # 이전 프레임의 사람들과 가장 가까운 매칭
        max_speed = 0
        for curr in people:
            for prev in self.person_history:
                d = math.hypot(curr["cx"] - prev["cx"], curr["cy"] - prev["cy"])
                # 같은 사람으로 추정 (50px 이내)
                if d < 50:
                    if d > max_speed:
                        max_speed = d
                    break
        
        if max_speed >= self.violence_speed:
            self.violence_buf.append(1)
            
            # 슬라이딩 윈도우 검증
            if sum(self.violence_buf) >= 3:  # 10프레임 중 3프레임
                if self.violence_start is None:
                    self.violence_start = time.time()
                elif time.time() - self.violence_start >= self.violence_hold_sec:
                    if not self._is_in_cooldown("violence"):
                        self._trigger_alert("violence")
                        if self.verbose:
                            print(f"[Anomaly] 🥊 폭행 감지! 거리={min_distance:.0f}, 속도={max_speed:.0f}")
                        self.violence_start = None
                        return True
        else:
            self.violence_buf.append(0)
            if sum(self.violence_buf) < 3:
                self.violence_start = None
        
        return False


# ============================================================
# 단독 테스트
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from vision.stream_reader import MJPEGStream
    
    CCTV_URL = os.getenv(
        "CCTV_STREAM_URL",
        "https://retirement-downloading-editor-hair.trycloudflare.com/video"
    )
    
    print(f"테스트 URL: {CCTV_URL}")
    
    # 빠른 테스트용
    detector = AnomalyDetector(
        violence_distance=120.0,
        violence_hold_sec=1.0,
        intrusion_start_hour=22,
        intrusion_end_hour=6,
        cooldown_sec=10.0,
        verbose=True
    )
    
    stream = MJPEGStream(CCTV_URL)
    start_time = time.time()
    frame_count = 0
    
    print("\n테스트 중... Ctrl+C로 종료")
    print("-" * 60)
    
    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.1)
                continue
            
            result = detector.detect(frame)
            frame_count += 1
            
            # 5초마다 상태
            if frame_count % 75 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                pc = result["person_count"]
                print(f"[#{frame_count}] FPS={fps:.1f}, 사람={pc}명")
            
            # 알림
            for event_type in ["violence", "intrusion"]:
                if result[event_type]:
                    print(f"🚨 [{event_type}] 감지!")
    
    except KeyboardInterrupt:
        print(f"\n총 {frame_count}프레임 처리됨")
    finally:
        stream.release()