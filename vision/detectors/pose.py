"""
vision/detectors/pose.py
YOLO-Pose 기반 자세 분석 (쓰러짐 + 장시간 체류)

기존 pose_detector.py의 로직을 모듈로 분리.
"""

import os
import time
import math
from collections import deque
from typing import Optional, Dict, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics 미설치. pip install ultralytics")


class PoseDetector:
    """YOLO-Pose 기반 쓰러짐 + 체류 감지기"""
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        # 쓰러짐 파라미터
        angle_th: float = 35.0,        # 몸통선 각도 임계값 (도)
        fall_frames: int = 5,           # deque 크기
        fall_hits: int = 3,             # deque 내 누움 판정 프레임 수
        fall_hold_sec: float = 5.0,    # 누움 지속 시간 (초)
        # 체류 파라미터
        loiter_radius: float = 40.0,    # 박스 중심 이동 허용 (픽셀)
        loiter_sec: float = 1800.0,     # 같은 위치 지속 시간 (초)
        cooldown_sec: float = 30.0,     # 알림 쿨다운
        # 기타
        device: str = "cuda",           # "cuda" or "cpu"
        verbose: bool = False,
    ):
        print(f"[PoseDetector] 모델 로딩: {model_path} (device={device})")
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            print("[PoseDetector] GPU 사용 불가, CPU로 진행")
        
        # 파라미터
        self.angle_th = angle_th
        self.fall_frames = fall_frames
        self.fall_hits = fall_hits
        self.fall_hold_sec = fall_hold_sec
        self.loiter_radius = loiter_radius
        self.loiter_sec = loiter_sec
        self.cooldown_sec = cooldown_sec
        self.verbose = verbose
        
        # 상태
        self.fall_buf = deque(maxlen=fall_frames)
        self.fall_start: Optional[float] = None
        self.loiter_anchor: Optional[tuple] = None  # (cx, cy, start_time)
        
        # 쿨다운
        self.last_fall_alert = 0.0
        self.last_loiter_alert = 0.0
        
        print("[PoseDetector] 준비 완료")
    
    @staticmethod
    def _trunk_angle(keypoints) -> float:
        """어깨-엉덩이 선의 수직축 대비 각도 (도)"""
        try:
            l_sh, r_sh = keypoints[5], keypoints[6]
            l_hp, r_hp = keypoints[11], keypoints[12]
            if (l_sh[2] < 0.3 or r_sh[2] < 0.3 or
                l_hp[2] < 0.3 or r_hp[2] < 0.3):
                return 90.0
            sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
            dx = hp[0] - sh[0]
            dy = hp[1] - sh[1]
            return abs(math.degrees(math.atan2(dx, dy)))
        except Exception:
            return 90.0
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 1장에 대해 자세 분석
        
        Returns:
            {
                "fall": bool,         # 쓰러짐 감지 여부
                "loitering": bool,     # 체류 감지 여부
                "person_detected": bool,
                "details": {...}
            }
        """
        result = {
            "fall": False,
            "loitering": False,
            "person_detected": False,
            "details": {}
        }
        
        # YOLO 추론
        results = self.model(frame, verbose=False)
        r = results[0]
        kp = r.keypoints
        bx = r.boxes
        
        person_down = False
        cx, cy = None, None
        angle = None
        
        if kp is not None and len(kp.data) > 0:
            result["person_detected"] = True
            k = kp.data[0].cpu().numpy()  # (17, 3)
            angle = self._trunk_angle(k)
            
            # 박스 중심 + 종횡비 (먼저 계산)
            box_aspect = 0.0
            if bx is not None and len(bx.xyxy) > 0:
                x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                box_w = x2 - x1
                box_h = y2 - y1
                box_aspect = box_w / max(box_h, 1)
            
            # 쓰러짐 판정: 각도 작음 + 종횡비 1.0 이상 (진짜 누움)
            if angle < self.angle_th and angle < 30 and box_aspect > 1.0:
                person_down = True
        
        result["details"]["angle"] = angle
        result["details"]["center"] = (cx, cy) if cx else None
        
        # 디버그용 진단
        if bx is not None and len(bx.xyxy) > 0:
            x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy()
            box_w = x2 - x1
            box_h = y2 - y1
            aspect = box_w / max(box_h, 1)
            result["details"]["aspect"] = aspect
            
            if self.verbose and result["person_detected"]:
                print(f"  [Pose 진단] 각도={angle:.1f}°, 종횡비={aspect:.2f}, "
                      f"박스(W={int(box_w)}, H={int(box_h)}), 누움판정={person_down}")

        # 디버그용: 박스 종횡비 확인
        if bx is not None and len(bx.xyxy) > 0:
            x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy()
            box_w = x2 - x1
            box_h = y2 - y1
            aspect = box_w / max(box_h, 1)
            result["details"]["aspect"] = aspect
            result["details"]["box_h"] = int(box_h)
            
            # 진단 출력 (사람 감지될 때만)
            if self.verbose and result["person_detected"]:
                print(f"  [Pose 진단] 각도={angle:.1f}°, 종횡비={aspect:.2f}, "
                      f"박스(W={int(box_w)}, H={int(box_h)}), 누움판정={person_down}")
        
        # ── 쓰러짐 판정 ────────────────────────────
        self.fall_buf.append(1 if person_down else 0)
        if sum(self.fall_buf) >= self.fall_hits:
            if self.fall_start is None:
                self.fall_start = time.time()
            elif time.time() - self.fall_start >= self.fall_hold_sec:
                # 쿨다운 확인
                if time.time() - self.last_fall_alert >= self.cooldown_sec:
                    result["fall"] = True
                    result["details"]["fall_duration"] = int(time.time() - self.fall_start)
                    result["details"]["fall_confidence"] = float(sum(self.fall_buf) / len(self.fall_buf))
                    self.last_fall_alert = time.time()
                    if self.verbose:
                        print(f"[Pose] 🚨 쓰러짐 감지!")
                self.fall_start = None
        else:
            self.fall_start = None
        
        # ── 장시간 체류 판정 ───────────────────────
        if cx is not None:
            if self.loiter_anchor is None:
                self.loiter_anchor = (cx, cy, time.time())
            else:
                ax, ay, t0 = self.loiter_anchor
                dist = math.hypot(cx - ax, cy - ay)
                if dist < self.loiter_radius:
                    duration = time.time() - t0
                    if duration >= self.loiter_sec:
                        if time.time() - self.last_loiter_alert >= self.cooldown_sec:
                            result["loitering"] = True
                            result["details"]["loitering_duration"] = int(duration)
                            self.last_loiter_alert = time.time()
                            if self.verbose:
                                print(f"[Pose] 🚨 장시간 체류 감지!")
                        self.loiter_anchor = (cx, cy, time.time())
                else:
                    self.loiter_anchor = (cx, cy, time.time())
        
        return result


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
    
    # 빠른 테스트를 위해 임계값 낮춤
    detector = PoseDetector(
        fall_hold_sec=2.0,   # 5초 → 2초
        loiter_sec=10.0,      # 1800초 → 10초
        cooldown_sec=5.0,
        verbose=True
    )
    
    stream = MJPEGStream(CCTV_URL)
    start_time = time.time()
    frame_count = 0
    
    print("\n테스트 중... Ctrl+C로 종료")
    print("쓰러짐 또는 체류 감지 시 알림 메시지 출력")
    print("-" * 60)
    
    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.1)
                continue
            
            result = detector.detect(frame)
            frame_count += 1
            
            # 5초마다 상태 출력
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                d = result["details"]
                person = "있음" if result["person_detected"] else "없음"
                angle = f"{d.get('angle'):.1f}°" if d.get('angle') else "N/A"
                print(f"[#{frame_count}] FPS={fps:.1f}, 사람={person}, 각도={angle}")
            
            # 알림
            if result["fall"]:
                print(f"🚨 [쓰러짐] 지속 {result['details']['fall_duration']}초")
            if result["loitering"]:
                print(f"🚨 [체류] 지속 {result['details']['loitering_duration']}초")
    
    except KeyboardInterrupt:
        print(f"\n총 {frame_count}프레임 처리됨")
    finally:
        stream.release()