"""
Roboflow에서 화재 감지 모델 다운로드

사용법:
1. API_KEY를 본인 Private API Key로 변경
2. python download_fire_model.py 실행
"""
import os
from roboflow import Roboflow

# ============================================================
# ⚠️ 본인 API KEY를 여기에 붙여넣기
# ============================================================
API_KEY = "Zs9Eh7xDt8GOJcC3YGP4"
# ============================================================

if API_KEY == "여기에_본인_API_KEY_붙여넣기" or not API_KEY:
    print("❌ API_KEY를 본인 Private API Key로 변경하세요!")
    print("   https://app.roboflow.com/settings/api 에서 복사")
    exit(1)

print("=" * 60)
print("Roboflow 화재 감지 모델 다운로드")
print("=" * 60)

# Roboflow 클라이언트 초기화
rf = Roboflow(api_key=API_KEY)

# anelinga 워크스페이스의 Fire-Detection-YOLOv8 프로젝트
print("\n[1/3] 프로젝트 접근 중...")
project = rf.workspace("anelinga").project("fire-detection-yolov8-ylrh2")
print(f"✅ 프로젝트: {project.name}")

# 버전 1
print("\n[2/3] 버전 1 선택...")
version = project.version(1)

# YOLOv8 형식으로 다운로드
print("\n[3/3] 모델 + 데이터셋 다운로드 중...")
print("(약 100~200MB, 1~3분 소요)")
dataset = version.download(
    "yolov8",
    location="models/fire_detection"
)

print("\n" + "=" * 60)
print("✅ 다운로드 완료!")
print(f"위치: {dataset.location}")
print("=" * 60)

# 다운로드 결과 확인
import glob
print("\n다운로드된 파일들:")
for f in glob.glob(f"{dataset.location}/**/*.pt", recursive=True):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  - {f} ({size_mb:.1f} MB)")

print("\n💡 다음 단계: detectors/fire.py 작성")