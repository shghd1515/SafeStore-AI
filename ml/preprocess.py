"""
01_preprocess.py
노트북 MariaDB에서 sensor_combined 테이블을 읽어
전처리 후 sensor_cleaned.csv로 저장

변경사항:
  - DB: 노트북 MariaDB (192.168.101.2, port 3307)
  - 테이블: sensor_combined (온습도 + 미세먼지 통합)
  - 이벤트 컬럼 포함
  - pm1, pm25, pm10 전처리 추가

실행:
    pip install pandas sqlalchemy pymysql python-dotenv
    python 01_preprocess.py
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── 0. DB 연결 ────────────────────────────────────────────────────────────────
def get_engine():
    supabase_url = os.getenv("SUPABASE_DB_URL")
    if supabase_url:
        return create_engine(supabase_url, pool_pre_ping=True)
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER','root01')}:{os.getenv('DB_PASSWORD','00000')}"
        f"@{os.getenv('DB_HOST','192.168.101.2')}:{os.getenv('DB_PORT','3307')}"
        f"/{os.getenv('DB_NAME','sensor_db')}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)


def load_raw_data(engine) -> pd.DataFrame:
    """
    sensor_combined 테이블에서 전체 데이터 로드
    컬럼: id, temperature, humidity, pm1, pm25, pm10, event, recorded_at
    """
    sql = """
        SELECT recorded_at, temperature, humidity,
               pm1, pm25, pm10, event
        FROM sensor_combined
        ORDER BY recorded_at ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, parse_dates=["recorded_at"])
    print(f"[로드] 총 {len(df):,} 행 읽기 완료")
    return df


# ── 1. 기본 정보 확인 ─────────────────────────────────────────────────────────
def check_basic_info(df: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print("[1] 기본 정보")
    print("="*50)
    print(f"  행 수       : {len(df):,}")
    print(f"  컬럼        : {df.columns.tolist()}")
    print(f"\n  첫 3행:\n{df.head(3)}")
    print(f"\n  마지막 3행:\n{df.tail(3)}")
    print(f"\n  최초 기록   : {df['recorded_at'].min()}")
    print(f"  최근 기록   : {df['recorded_at'].max()}")
    print(f"  총 기간     : {df['recorded_at'].max() - df['recorded_at'].min()}")
    print(f"\n  결측값:\n{df.isnull().sum()}")

    # 이벤트 현황
    events = df[df['event'].notna()]['event'].value_counts()
    if not events.empty:
        print(f"\n  이벤트 현황:\n{events}")
    else:
        print("\n  이벤트: 없음")


# ── 2. 결측값 처리 ────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*50)
    print("[2] 결측값 처리")
    print("="*50)
    # PM2.5 null 행 제거 (센서 미연결 기간 데이터 제외)
    before = len(df)
    df = df[df["pm25"].notna() & (df["pm25"] > 0)]
    print(f"  PM2.5 없는 행 제거: {before - len(df)}행 제거 → {len(df)}행 남음")

    missing_before = df[['temperature', 'humidity', 'pm1', 'pm25', 'pm10']].isnull().sum().sum()
    print(f"  처리 전 결측값 총계: {missing_before}")

    df = df.set_index("recorded_at").sort_index()

    # 온습도 보간
    for col in ["temperature", "humidity"]:
        df[col] = df[col].interpolate(method="time").ffill().bfill()

    # 미세먼지 보간 (pm 데이터는 나중에 추가됐으므로 앞쪽 NULL은 0으로)
    for col in ["pm1", "pm25", "pm10"]:
        df[col] = df[col].interpolate(method="time").ffill().bfill().fillna(0)

    df = df.reset_index()
    missing_after = df[['temperature', 'humidity', 'pm1', 'pm25', 'pm10']].isnull().sum().sum()
    print(f"  처리 후 결측값 총계: {missing_after}")
    return df


# ── 3. 중복 제거 ──────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*50)
    print("[3] 중복 제거")
    print("="*50)
    before = len(df)
    df = df.drop_duplicates(subset=["recorded_at"]).reset_index(drop=True)
    print(f"  제거된 중복 행: {before - len(df)}개")
    return df


# ── 4. 이상치 제거 ────────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*50)
    print("[4] 이상치 제거")
    print("="*50)

    before = len(df)

    # 물리적 범위 필터
    physical_mask = (
        df["temperature"].between(0.0, 50.0) &
        df["humidity"].between(10.0, 100.0) &
        df["pm25"].between(0, 1000) &
        df["pm10"].between(0, 1000)
    )
    df = df[physical_mask].copy()
    after_physical = len(df)
    print(f"  [물리 범위 필터] 제거: {before - after_physical}행")

    # IQR 필터
    def iqr_filter(series: pd.Series, k: float = 2.0) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        print(f"    {series.name}: [{lower:.2f}, {upper:.2f}]")
        return (series >= lower) & (series <= upper)

    iqr_mask = (
        iqr_filter(df["temperature"]) &
        iqr_filter(df["humidity"]) &
        iqr_filter(df["pm25"])
    )
    df = df[iqr_mask].copy().reset_index(drop=True)
    print(f"  [IQR 필터]      제거: {after_physical - len(df)}행")
    print(f"  최종 남은 행   : {len(df):,} / {before:,}")
    return df


# ── 5. 피처 엔지니어링 ────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*50)
    print("[5] 피처 엔지니어링 (강화)")
    print("="*50)

    dt = pd.to_datetime(df["recorded_at"])

    # 기본 시간 파생변수
    df["hour"]        = dt.dt.hour
    df["minute"]      = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = dt.dt.month
    df["date"]        = dt.dt.date

    # 주기적 인코딩
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # 시간대 그룹 (새벽·아침·낮·저녁·밤)
    def get_time_group(h):
        if 0  <= h <  6: return 0  # 새벽
        if 6  <= h < 10: return 1  # 아침
        if 10 <= h < 14: return 2  # 낮
        if 14 <= h < 18: return 3  # 오후
        if 18 <= h < 22: return 4  # 저녁
        return 5                   # 밤
    df["time_group"] = df["hour"].apply(get_time_group)

    # 이동 평균 (다양한 구간)
    for window in [10, 30, 60, 120]:
        df[f"temp_ma{window}"] = df["temperature"].rolling(window, min_periods=1).mean()
        df[f"humi_ma{window}"] = df["humidity"].rolling(window, min_periods=1).mean()
        df[f"pm25_ma{window}"] = df["pm25"].rolling(window, min_periods=1).mean()

    # 변화량 (1분·5분·10분)
    df["temp_diff"]    = df["temperature"].diff(1).fillna(0)
    df["humi_diff"]    = df["humidity"].diff(1).fillna(0)
    df["pm25_diff"]    = df["pm25"].diff(1).fillna(0)
    df["temp_diff5"]   = df["temperature"].diff(5).fillna(0)
    df["humi_diff5"]   = df["humidity"].diff(5).fillna(0)
    df["pm25_diff5"]   = df["pm25"].diff(5).fillna(0)
    df["temp_diff10"]  = df["temperature"].diff(10).fillna(0)
    df["humi_diff10"]  = df["humidity"].diff(10).fillna(0)
    df["pm25_diff10"]  = df["pm25"].diff(10).fillna(0)

    # 변화 속도 (가속도)
    df["temp_accel"]  = df["temp_diff"].diff(1).fillna(0)
    df["humi_accel"]  = df["humi_diff"].diff(1).fillna(0)
    df["pm25_accel"]  = df["pm25_diff"].diff(1).fillna(0)

    # 롤링 표준편차 (변동성)
    df["temp_std10"]  = df["temperature"].rolling(10, min_periods=1).std().fillna(0)
    df["humi_std10"]  = df["humidity"].rolling(10, min_periods=1).std().fillna(0)
    df["pm25_std10"]  = df["pm25"].rolling(10, min_periods=1).std().fillna(0)

    # 교차 피처
    df["temp_humi"]   = df["temperature"] * df["humidity"] / 100
    df["pm25_hour"]   = df["pm25"] * df["hour_sin"]

    # 이벤트 인코딩
    df["has_event"]   = df["event"].notna().astype(int)

    print(f"  추가된 컬럼 수: {df.shape[1]}개")
    print(f"  컬럼 목록: {df.columns.tolist()}")
    return df


# ── 6. 기술 통계 출력 ─────────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print("[6] 전처리 완료 후 기술 통계")
    print("="*50)
    cols = ["temperature", "humidity", "pm1", "pm25", "pm10"]
    print(df[cols].describe().round(2).to_string())


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("===== 전처리 시작 =====\n")

    engine = get_engine()
    df = load_raw_data(engine)

    check_basic_info(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = add_features(df)
    print_stats(df)

    output_path = "sensor_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"\n전처리 완료! 저장: {output_path}  ({len(df):,} 행)")


if __name__ == "__main__":
    main()
