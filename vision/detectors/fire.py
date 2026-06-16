"""
vision/detectors/fire.py
YOLOv8/v10 기반 화재 + 연기 감지

사전 학습된 모델 사용 (luminous0219/fire-and-smoke-detection-yolov8)
"""

import os
import time
from typing import Dict, Any
from collections import deque

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics 미설치. pip install ultralytics")


class FireDetector:
    """화재 + 연기 감지기"""
    
    def __init__(
        self,
        model_path: str = "models/fire_detection/best.pt",
        confidence_threshold: float = 0.5,       # 모델 자체 임계값
        hit_threshold: int = 3,                   # N프레임 중 X프레임 이상 감지
        window_size: int = 5,                     # 슬라이딩 윈도우 크기
        hold_sec: float = 2.0,                    # 지속 시간 임계값 (초)
        cooldown_sec: float = 60.0,               # 알림 쿨다운 (1분)
        device: str = "cuda",
        verbose: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        
        print(f"[FireDetector] 모델 로딩: {model_path} (device={device})")
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except Exception:
            print("[FireDetector] GPU 사용 불가, CPU로 진행")
        
        # 클래스 정보 출력
        self.classes = self.model.names
        print(f"[FireDetector] 클래스: {self.classes}")
        
        # 파라미터
        self.conf_th = confidence_threshold
        self.hit_threshold = hit_threshold
        self.window_size = window_size
        self.hold_sec = hold_sec
        self.cooldown_sec = cooldown_sec
        self.verbose = verbose
        
        # 상태
        self.fire_buf = deque(maxlen=window_size)   # 화재 감지 슬라이딩 윈도우
        self.smoke_buf = deque(maxlen=window_size)
        self.fire_start = None    # 화재 지속 시작 시간
        self.smoke_start = None
        self.last_fire_alert = 0.0
        self.last_smoke_alert = 0.0
        
        print("[FireDetector] 준비 완료")
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """프레임 1장에 대해 화재/연기 감지
        
        Returns:
            {
                "fire_detected": bool,
                "smoke_detected": bool,
                "fire_confidence": float,
                "smoke_confidence": float,
                "bbox": list or None,
                "details": {...}
            }
        """
        result = {
            "fire_detected": False,
            "smoke_detected": False,
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0,
            "bbox": None,
            "details": {}
        }
        
        # YOLO 추론
        try:
            results = self.model(frame, verbose=False, conf=self.conf_th)
        except Exception as e:
            print(f"[FireDetector] 추론 오류: {e}")
            return result
        
        r = results[0]
        
        # 감지 결과 파싱
        fire_found = False
        smoke_found = False
        max_fire_conf = 0.0
        max_smoke_conf = 0.0
        best_bbox = None
        
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.classes[cls_id].lower()
                
                if "fire" in cls_name or cls_name == "flame":
                    fire_found = True
                    if conf > max_fire_conf:
                        max_fire_conf = conf
                        best_bbox = box.xyxy[0].cpu().numpy().tolist()
                elif "smoke" in cls_name:
                    smoke_found = True
                    if conf > max_smoke_conf:
                        max_smoke_conf = conf
        
        result["fire_confidence"] = max_fire_conf
        result["smoke_confidence"] = max_smoke_conf
        result["bbox"] = best_bbox
        
        # ── 화재 판정 (슬라이딩 윈도우) ─────────────
        self.fire_buf.append(1 if fire_found else 0)
        if sum(self.fire_buf) >= self.hit_threshold:
            if self.fire_start is None:
                self.fire_start = time.time()
            elif time.time() - self.fire_start >= self.hold_sec:
                # 쿨다운 확인
                if time.time() - self.last_fire_alert >= self.cooldown_sec:
                    result["fire_detected"] = True
                    result["details"]["fire_duration"] = int(time.time() - self.fire_start)
                    self.last_fire_alert = time.time()
                    if self.verbose:
                        print(f"[Fire] 🔥 화재 감지! 신뢰도={max_fire_conf:.2f}")
                self.fire_start = None
        else:
            self.fire_start = None
        
        # ── 연기 판정 ───────────────────────────────
        self.smoke_buf.append(1 if smoke_found else 0)
        if sum(self.smoke_buf) >= self.hit_threshold:
            if self.smoke_start is None:
                self.smoke_start = time.time()
            elif time.time() - self.smoke_start >= self.hold_sec:
                if time.time() - self.last_smoke_alert >= self.cooldown_sec:
                    result["smoke_detected"] = True
                    result["details"]["smoke_duration"] = int(time.time() - self.smoke_start)
                    self.last_smoke_alert = time.time()
                    if self.verbose:
                        print(f"[Fire] 💨 연기 감지! 신뢰도={max_smoke_conf:.2f}")
                self.smoke_start = None
        else:
            self.smoke_start = None
        
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
    
    # 빠른 테스트용 임계값 낮춤
    detector = FireDetector(
        confidence_threshold=0.4,
        hit_threshold=2,
        hold_sec=1.0,
        cooldown_sec=10.0,
        verbose=True
    )
    
    stream = MJPEGStream(CCTV_URL)
    start_time = time.time()
    frame_count = 0
    
    print("\n테스트 중... Ctrl+C로 종료")
    print("화재 영상 또는 라이터 등 비춰보세요")
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
            if frame_count % 75 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                fc = result["fire_confidence"]
                sc = result["smoke_confidence"]
                print(f"[#{frame_count}] FPS={fps:.1f}, 화재신뢰도={fc:.2f}, 연기신뢰도={sc:.2f}")
            
            # 알림
            if result["fire_detected"]:
                print(f"🔥 [화재] 지속 {result['details']['fire_duration']}초, 신뢰도={result['fire_confidence']:.2f}")
            if result["smoke_detected"]:
                print(f"💨 [연기] 지속 {result['details']['smoke_duration']}초, 신뢰도={result['smoke_confidence']:.2f}")
    
    except KeyboardInterrupt:
        print(f"\n총 {frame_count}프레임 처리됨")
    finally:
        stream.release()