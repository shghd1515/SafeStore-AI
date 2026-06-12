"""
vision/stream_proxy.py
라즈베리파이2호로부터 영상을 받아서 브라우저로 MJPEG 스트림으로 송출하는 프록시.

구조:
  [라파이2호 :9999] ──TCP(JPEG+struct 'Q')──> [stream_proxy] ──MJPEG──> [브라우저 <img>]

특징:
  - 백그라운드 스레드로 라파이에 연결 유지
  - 최신 프레임만 메모리에 보관 (오래된 프레임 버림)
  - 라파이 연결 끊겨도 서버는 살아있음 (자동 재연결 시도)
  - 라파이 미가동 시 '연결 대기 중' 플레이스홀더 프레임 송출

사용:
  from vision.stream_proxy import start_proxy, mjpeg_generator
  start_proxy()  # 백그라운드 시작
  # FastAPI 엔드포인트:
  #   @app.get('/video-stream')
  #   def stream():
  #       return StreamingResponse(mjpeg_generator(),
  #                                media_type='multipart/x-mixed-replace; boundary=frame')
"""

import os
import sys
import time
import socket
import struct
import threading
from io import BytesIO

# OpenCV / numpy (라파이 영상 디코드용)
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[stream_proxy] opencv-python 미설치 — 플레이스홀더만 사용")

# 환경변수에서 라파이 정보 읽기
PI_HOST = os.getenv("PI_STREAM_HOST", "192.168.14.11")
PI_PORT = int(os.getenv("PI_STREAM_PORT", "9999"))
RECONNECT_INTERVAL = 5  # 연결 실패 시 재시도 간격 (초)

# 전역 상태
_latest_jpeg: bytes = b""           # 가장 최근 JPEG 프레임 (bytes)
_latest_lock = threading.Lock()
_connected = False
_running = False
_thread: threading.Thread | None = None


def _make_placeholder_jpeg(text: str = "연결 대기 중") -> bytes:
    """라파이 미연결 시 사용할 플레이스홀더 이미지 (다크 + 메시지)."""
    if not HAS_CV2:
        # opencv 없으면 1x1 검은 JPEG (최소 구색)
        return (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00' + b'\x08' * 64
            + b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x14\x00\x01' + b'\x00' * 15 + b'\x00'
            b'\xff\xc4\x00\x14\x10\x01' + b'\x00' * 15 + b'\x00'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfa\xff\xd9'
        )
    # 640x480 다크 네이비 배경
    img = np.full((480, 640, 3), (41, 25, 10), dtype=np.uint8)  # BGR
    # 가운데 메시지
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_full = f"CCTV: {text}"
    text_size = cv2.getTextSize(text_full, font, 0.8, 2)[0]
    x = (640 - text_size[0]) // 2
    y = (480 + text_size[1]) // 2
    cv2.putText(img, text_full, (x, y), font, 0.8, (200, 200, 200), 2)
    # 시각
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, ts, (10, 470), font, 0.5, (150, 150, 150), 1)
    # JPEG 인코딩
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _receive_loop():
    """백그라운드 스레드: 라파이에 연결해서 프레임 계속 받기"""
    global _latest_jpeg, _connected, _running

    payload_size = struct.calcsize("Q")
    placeholder = _make_placeholder_jpeg("라파이 연결 대기 중")

    while _running:
        sock = None
        try:
            print(f"[stream_proxy] 라파이 연결 시도: {PI_HOST}:{PI_PORT}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((PI_HOST, PI_PORT))
            sock.settimeout(None)
            _connected = True
            print(f"[stream_proxy] 라파이 연결 성공")

            data = b""
            while _running:
                # 패킷 헤더(크기 정보) 수신
                while len(data) < payload_size:
                    packet = sock.recv(4 * 1024)
                    if not packet:
                        raise ConnectionError("라파이 연결 종료")
                    data += packet
                msg_size = struct.unpack("Q", data[:payload_size])[0]
                data = data[payload_size:]

                # JPEG 본체 수신
                while len(data) < msg_size:
                    packet = sock.recv(4 * 1024)
                    if not packet:
                        raise ConnectionError("라파이 연결 종료")
                    data += packet
                frame_bytes = data[:msg_size]
                data = data[msg_size:]

                # 최신 프레임만 보관 (오래된 건 자동 폐기)
                with _latest_lock:
                    _latest_jpeg = frame_bytes

        except (socket.timeout, ConnectionError, ConnectionRefusedError, OSError) as e:
            _connected = False
            with _latest_lock:
                _latest_jpeg = placeholder
            print(f"[stream_proxy] 연결 실패/끊김: {e}. {RECONNECT_INTERVAL}초 후 재시도")
        except Exception as e:
            _connected = False
            print(f"[stream_proxy] 예외: {e}")
        finally:
            if sock:
                try: sock.close()
                except: pass

        if _running:
            time.sleep(RECONNECT_INTERVAL)


def start_proxy():
    """프록시 백그라운드 스레드 시작."""
    global _running, _thread, _latest_jpeg
    if _thread and _thread.is_alive():
        return
    _running = True
    with _latest_lock:
        _latest_jpeg = _make_placeholder_jpeg("초기화 중")
    _thread = threading.Thread(target=_receive_loop, daemon=True, name="stream-proxy")
    _thread.start()
    print("[stream_proxy] 백그라운드 시작됨")


def stop_proxy():
    global _running
    _running = False


def get_latest_jpeg() -> bytes:
    """현재 최신 프레임 JPEG bytes 반환."""
    with _latest_lock:
        return _latest_jpeg


def is_connected() -> bool:
    return _connected


def mjpeg_generator(target_fps: int = 15):
    """
    FastAPI StreamingResponse용 제너레이터.
    multipart/x-mixed-replace 형식으로 JPEG 프레임을 연속 송출.
    """
    interval = 1.0 / max(1, target_fps)
    while True:
        frame = get_latest_jpeg()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )
        time.sleep(interval)


if __name__ == "__main__":
    # 단독 테스트 (라파이 켜져있으면 실시간 프레임 받아짐)
    start_proxy()
    print("Ctrl+C로 종료. 5초마다 상태 출력...")
    try:
        while True:
            time.sleep(5)
            jpeg = get_latest_jpeg()
            print(f"[status] connected={_connected}, frame_size={len(jpeg)} bytes")
    except KeyboardInterrupt:
        stop_proxy()
        print("종료")
