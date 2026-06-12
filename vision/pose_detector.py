"""
vision/pose_detector.py
YOLO-Pose 기반 무인매장 이상행동 감지 (노트북 GPU에서 실행)

기능:
  · 쓰러짐 감지 (Fall): 어깨-엉덩이 몸통선 각도 < 30°가 5초 이상 지속
  · 장시간 체류 (Loitering): 박스 중심이 40px 내에 1800초 이상 머무름

기존 노트북 코드(작업폴더: backend/pose_detector.py)에서 옮겨온 통합 버전.
이상 감지 시 vision.anomaly_uploader.push_event() 호출.

실행:
    python -m vision.pose_detector
    (라즈베리파이2호 stream.py가 192.168.14.11:9999 송출 중이어야 함)
"""

import os
import sys
import time
import math
import struct
import socket
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv

# YOLO (ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    print("[!] ultralytics 미설치. pip install ultralytics")
    sys.exit(1)

# 같은 패키지의 업로더
from vision.anomaly_uploader import push_event

load_dotenv()

# ── 설정 ────────────────────────────────────────────────────────────────────
PI_HOST       = os.getenv("PI_STREAM_HOST", "192.168.14.11")
PI_PORT       = int(os.getenv("PI_STREAM_PORT", "9999"))
MODEL_PATH    = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")

ANGLE_TH      = 35       # 몸통선 각도 임계값 (degrees)
FALL_FRAMES   = 5        # deque 길이
FALL_HITS     = 3        # deque 내 누움 판정 프레임 수
FALL_HOLD_SEC = 5        # 누움이 N초 연속 → fall 알림
LOITER_RADIUS = 40       # 박스 중심 이동 허용 (px)
LOITER_SEC    = 1800     # 같은 위치 N초 머무르면 loitering
COOLDOWN_SEC  = 30


def trunk_angle(keypoints) -> float:
    """어깨(평균)-엉덩이(평균) 선의 수직축 대비 각도를 도(°)로 반환"""
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
        # 수직축 대비 (atan2)
        return abs(math.degrees(math.atan2(dx, dy)))
    except Exception:
        return 90.0


def recv_frame(sock, payload_size=struct.calcsize("Q")):
    """라즈베리파이 stream.py가 JPEG+struct 'Q' 패킷으로 송출하는 프레임 수신"""
    data = b""
    while len(data) < payload_size:
        packet = sock.recv(4096)
        if not packet:
            return None
        data += packet
    msg_size = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]
    while len(data) < msg_size:
        data += sock.recv(4096)
    frame_bytes = data[:msg_size]
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    return frame


def main():
    print(f"[YOLO-Pose] 모델 로딩: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"[연결] {PI_HOST}:{PI_PORT} 시도...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_HOST, PI_PORT))
    print("[연결] OK")

    fall_buf = deque(maxlen=FALL_FRAMES)
    fall_start = None
    loiter_anchor = None  # (cx, cy, start_time)

    try:
        while True:
            frame = recv_frame(sock)
            if frame is None:
                print("[수신] 연결 종료")
                break

            results = model(frame, verbose=False)
            r = results[0]
            kp = r.keypoints
            bx = r.boxes

            person_down = False
            cx, cy = None, None

            if kp is not None and len(kp.data) > 0:
                k = kp.data[0].cpu().numpy()  # (17, 3)
                ang = trunk_angle(k)
                if ang < ANGLE_TH and ang < 30:
                    person_down = True

                if bx is not None and len(bx.xyxy) > 0:
                    x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # ── 쓰러짐 판정 ────────────────────────────
            fall_buf.append(1 if person_down else 0)
            if sum(fall_buf) >= FALL_HITS:
                if fall_start is None:
                    fall_start = time.time()
                elif time.time() - fall_start >= FALL_HOLD_SEC:
                    push_event(
                        event_type="fall",
                        duration_sec=int(time.time() - fall_start),
                        confidence=float(sum(fall_buf) / len(fall_buf)),
                        cooldown=COOLDOWN_SEC,
                    )
                    fall_start = None  # 알림 후 리셋
            else:
                fall_start = None

            # ── 장시간 체류 판정 ───────────────────────
            if cx is not None:
                if loiter_anchor is None:
                    loiter_anchor = (cx, cy, time.time())
                else:
                    ax, ay, t0 = loiter_anchor
                    dist = math.hypot(cx - ax, cy - ay)
                    if dist < LOITER_RADIUS:
                        if time.time() - t0 >= LOITER_SEC:
                            push_event(
                                event_type="loitering",
                                duration_sec=int(time.time() - t0),
                                confidence=0.75,
                                cooldown=COOLDOWN_SEC,
                            )
                            loiter_anchor = (cx, cy, time.time())  # 재시작
                    else:
                        loiter_anchor = (cx, cy, time.time())

            # 디버그 화면 (옵션)
            if os.getenv("YOLO_SHOW", "0") == "1":
                annotated = r.plot()
                cv2.imshow("SafeStore Pose", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[종료] 사용자 중단")
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
