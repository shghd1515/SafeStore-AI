"""
vision/detect_all.py
SafeStore AI 통합 이상행동 감지 시스템

여러 AI 모델을 동시에 실행:
  - YOLO-Pose: 쓰러짐, 장시간 체류
  - YOLOv8 Fire: 화재/연기 감지 (Day 1 추가 예정)
  - VideoMAE: 폭행/싸움 감지 (Day 2 추가 예정)

실행:
    python -m vision.detect_all
"""

import os
import sys
import time
import signal
from typing import Optional

from dotenv import load_dotenv

# 우리 모듈
from vision.stream_reader import MJPEGStream
from vision.detectors.pose import PoseDetector
from vision.detectors.fire import FireDetector
from vision.detectors.anomaly import AnomalyDetector  # ← 추가
from vision.anomaly_uploader import push_event

load_dotenv()


class SafeStoreDetector:
    """SafeStore AI 통합 감지 시스템"""
    
    def __init__(self, cctv_url: str):
        self.cctv_url = cctv_url
        self.stream: Optional[MJPEGStream] = None
        
        # 감지 모듈들 (나중에 fire, violence 추가)
        self.pose: Optional[PoseDetector] = None
        self.fire: Optional[FireDetector] = None        # Day 1 후반
        self.anomaly: Optional[AnomalyDetector] = None  # ← 추가
        
        self.running = False
        self.frame_count = 0
        self.start_time = 0
        
        # 통계
        self.stats = {
            "fall_count": 0,
            "loiter_count": 0,
            "fire_count": 0,
            "violence_count": 0,
            "intrusion_count": 0,
        }
    
    def setup(self):
        """모든 모듈 초기화"""
        print("=" * 60)
        print("SafeStore AI 통합 감지 시스템 시작")
        print("=" * 60)
        
        # 1. 영상 스트림
        print("\n[1/4] 영상 스트림 연결...")
        self.stream = MJPEGStream(self.cctv_url)
        
        # 2. YOLO-Pose
        print("\n[2/4] YOLO-Pose 로드...")
        self.pose = PoseDetector(
            angle_th=20.0,
            fall_hold_sec=10.0,      # 실전: 5초
            loiter_sec=60.0,         # 실전: 60초 (테스트로 1분)
            cooldown_sec=120.0,
            verbose=False,
        )
        
        # 3. 화재 감지
        print("\n[3/4] 화재 감지 로드...")
        self.fire = FireDetector(
            model_path="models/fire_detection/best.pt",
            confidence_threshold=0.5,
            hit_threshold=3,
            hold_sec=2.0,
            cooldown_sec=60.0,
            verbose=False,
        )

        # 4. 종합 이상행동 감지
        print("\n[4/4] 종합 이상행동 감지 로드...")
        self.anomaly = AnomalyDetector(
            violence_distance=120.0,
            violence_hold_sec=1.5,
            intrusion_start_hour=22,    # 22시~06시 비영업
            intrusion_end_hour=6,
            cooldown_sec=60.0,
            verbose=False,
        )
        
        print("\n✅ 모든 모듈 준비 완료")
        print("=" * 60)
    
    def _save_event(self, event_type: str, details: dict):
        """이상 이벤트를 Supabase에 저장"""
        try:
            push_event(
                event_type=event_type,
                duration_sec=details.get("duration", 0),
                confidence=details.get("confidence", 0.8),
                cooldown=30,
            )
            # 통계 키 매핑
            stat_key_map = {
                "fall": "fall_count",
                "loitering": "loiter_count",
                "fire": "fire_count",
                "violence": "violence_count",
                "running": "running_count",
                "crowding": "crowding_count",
                "intrusion": "intrusion_count",
            }
            stat_key = stat_key_map.get(event_type, f"{event_type}_count")
            if stat_key in self.stats:
                self.stats[stat_key] += 1
            print(f"📤 [Supabase 저장] {event_type}")
        except Exception as e:
            print(f"[Supabase 오류] {e}")
    
    def run(self):
        """메인 추론 루프"""
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Ctrl+C 핸들러
        def signal_handler(sig, frame):
            print("\n중단 신호 받음...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"\n[감지 시작] {self.cctv_url}")
        print("종료: Ctrl+C")
        print("-" * 60)
        
        try:
            while self.running:
                # 1. 프레임 수신
                frame = self.stream.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # 2. Pose 감지
                pose_result = self.pose.detect(frame)
                
                # 3. 결과 처리
                if pose_result["fall"]:
                    print(f"🚨 [쓰러짐] 지속 {pose_result['details']['fall_duration']}초")
                    self._save_event("fall", {
                        "duration": pose_result["details"]["fall_duration"],
                        "confidence": pose_result["details"]["fall_confidence"],
                    })
                
                if pose_result["loitering"]:
                    print(f"🚨 [체류] 지속 {pose_result['details']['loitering_duration']}초")
                    self._save_event("loitering", {
                        "duration": pose_result["details"]["loitering_duration"],
                        "confidence": 0.75,
                    })
                
                # 4. 화재 감지
                if self.fire:
                    fire_result = self.fire.detect(frame)
                    
                    if fire_result["fire_detected"]:
                        conf = fire_result["fire_confidence"]
                        duration = fire_result["details"].get("fire_duration", 0)
                        print(f"🔥 [화재] 지속 {duration}초, 신뢰도={conf:.2f}")
                        self._save_event("fire", {
                            "duration": duration,
                            "confidence": conf,
                        })
                    
                    if fire_result["smoke_detected"]:
                        conf = fire_result["smoke_confidence"]
                        duration = fire_result["details"].get("smoke_duration", 0)
                        print(f"💨 [연기] 지속 {duration}초, 신뢰도={conf:.2f}")
                        self._save_event("fire", {
                            "duration": duration,
                            "confidence": conf,
                        })
                
                
                # 5. 종합 이상행동 감지
                if self.anomaly:
                    anomaly_result = self.anomaly.detect(frame)
                    
                    if anomaly_result["violence"]:
                        print(f"🚨 [폭행/싸움] 감지!")
                        self._save_event("violence", {
                            "duration": 2,
                            "confidence": 0.85,
                        })
                    
                    
                    if anomaly_result["intrusion"]:
                        print(f"🚨 [비영업 시간 침입] 감지!")
                        self._save_event("intrusion", {
                            "duration": 0,
                            "confidence": 0.95,
                        })
               
                if self.frame_count % 150 == 0:
                    self._print_status()
        
        except Exception as e:
            print(f"[오류] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def _print_status(self):
        """현재 상태 출력"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"[상태] 프레임={self.frame_count}, FPS={fps:.1f}, "
              f"쓰러짐={self.stats['fall_count']}, "
              f"체류={self.stats['loiter_count']}, "
              f"화재={self.stats['fire_count']}, "
              f"폭행={self.stats['violence_count']}, "
              f"침입={self.stats['intrusion_count']}")
    
    def cleanup(self):
        """리소스 정리"""
        print("\n[정리] 리소스 해제 중...")
        if self.stream:
            self.stream.release()
        
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("실행 요약")
        print("=" * 60)
        print(f"총 실행 시간: {elapsed:.1f}초")
        print(f"처리 프레임: {self.frame_count}")
        print(f"평균 FPS: {self.frame_count / elapsed:.1f}" if elapsed > 0 else "")
        print(f"감지 이벤트:")
        print(f"  - 쓰러짐: {self.stats['fall_count']}회")
        print(f"  - 체류: {self.stats['loiter_count']}회")
        print(f"  - 화재: {self.stats['fire_count']}회")
        print(f"  - 폭행: {self.stats['violence_count']}회")
        print(f"  - 침입: {self.stats['intrusion_count']}회")
        print("=" * 60)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    CCTV_URL = os.getenv(
        "CCTV_STREAM_URL",
        "https://retirement-downloading-editor-hair.trycloudflare.com/video"
    )
    
    detector = SafeStoreDetector(CCTV_URL)
    detector.setup()
    detector.run()