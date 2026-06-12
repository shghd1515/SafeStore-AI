# Phase 1 적용 방법

## 파일 배치
- `start_server.bat` → `safestore_backend\start_server.bat`
- `SafeStore_바로가기.bat` → 바탕화면 (또는 어디든)
- `vision\stream_proxy.py` → `safestore_backend\vision\stream_proxy.py`
- `frontend\dashboard.html` → `safestore_backend\frontend\dashboard.html` (덮어쓰기)
- `main_patch.txt` → main.py 수정 가이드 (텍스트만 보면 됨)

## main.py 수정 (수동)
main_patch.txt 안의 코드를 main.py 두 곳에 추가:
1. import 영역 (상단)
2. /anomaly-events 엔드포인트 아래쪽

## 테스트
1. start_server.bat 더블클릭
2. 5초 뒤 브라우저 자동 실행
3. 보안 섹션의 CCTV 영역 확인
