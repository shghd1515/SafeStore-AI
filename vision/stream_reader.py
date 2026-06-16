"""
vision/stream_reader.py
Cloudflare Tunnel을 통한 MJPEG 영상 스트림 수신

사용 예:
    from vision.stream_reader import MJPEGStream
    
    stream = MJPEGStream("https://xxx.trycloudflare.com/video")
    while True:
        frame = stream.read()
        if frame is None:
            continue
        # frame 사용...
"""

import cv2
import time
import threading
from typing import Optional
import numpy as np


class MJPEGStream:
    """MJPEG 스트림에서 프레임을 읽어오는 클래스
    
    OpenCV의 VideoCapture를 사용하여 MJPEG 스트림에 접속.
    연결 끊김 시 자동 재연결 지원.
    """
    
    def __init__(self, url: str, reconnect_delay: float = 5.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.last_frame_time = 0
        self.frame_count = 0
        self._connect()
    
    def _connect(self):
        """스트림 연결 시도"""
        print(f"[MJPEGStream] 연결 시도: {self.url}")
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.url)
        
        # 버퍼 크기 최소화 (실시간 처리)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            print("[MJPEGStream] 연결 성공")
            return True
        else:
            print("[MJPEGStream] 연결 실패")
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """프레임 1장 읽기
        
        Returns:
            np.ndarray: 영상 프레임 (BGR), 실패 시 None
        """
        if self.cap is None or not self.cap.isOpened():
            print("[MJPEGStream] 재연결 중...")
            time.sleep(self.reconnect_delay)
            self._connect()
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[MJPEGStream] 프레임 읽기 실패, 재연결...")
            self._connect()
            return None
        
        self.last_frame_time = time.time()
        self.frame_count += 1
        return frame
    
    def get_stats(self):
        """스트림 통계"""
        return {
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time,
            "is_open": self.cap is not None and self.cap.isOpened()
        }
    
    def release(self):
        """스트림 종료"""
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================
# 단독 실행 시: 영상이 잘 받아지는지 테스트
# ============================================================
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Cloudflare URL (환경변수 또는 기본값)
    CCTV_URL = os.getenv(
        "CCTV_STREAM_URL",
        "https://retirement-downloading-editor-hair.trycloudflare.com/video"
    )
    
    print(f"테스트 URL: {CCTV_URL}")
    print("Ctrl+C로 종료")
    print("-" * 60)
    
    stream = MJPEGStream(CCTV_URL)
    start_time = time.time()
    
    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # 5초마다 통계 출력
            if stream.frame_count % 150 == 0 and stream.frame_count > 0:
                elapsed = time.time() - start_time
                fps = stream.frame_count / elapsed if elapsed > 0 else 0
                h, w = frame.shape[:2]
                print(f"✅ 프레임 #{stream.frame_count}, 크기: {w}x{h}, 평균 FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        fps = stream.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n[종료] 총 {stream.frame_count}프레임 수신, 평균 FPS: {fps:.1f}")
    finally:
        stream.release()
        print("스트림 해제 완료")